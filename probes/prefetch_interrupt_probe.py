#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decisive experiment: is `batch_is_exist` really the slow part, or is the
transfer simply being TRUNCATED?

Design goals:
  * exists_latency == get_latency  -> removes any assumption about which
    mooncake call is intrinsically more expensive. Both cost the same.
  * The only variable is WHEN the scheduler makes its first
    check_prefetch_progress() call (i.e. when it gives up on the operation).

Real sglang code (v0.5.9) is exercised; only the mooncake C++ object is
replaced by a latency-injecting stand-in.
"""

import queue
import threading
import time

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

LAT = 0.002  # 2 ms for BOTH batch_is_exist and batch_get_into


class FakeStore:
    def __init__(self):
        self.exists_calls = self.get_calls = 0
        self.exists_keys = self.get_keys = 0

    def batch_is_exist(self, key_strs):
        self.exists_calls += 1
        self.exists_keys += len(key_strs)
        time.sleep(LAT)
        return [1] * len(key_strs)

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        self.get_calls += 1
        self.get_keys += len(key_strs)
        time.sleep(LAT)
        return [max(1, int(s)) for s in buffer_sizes]


class FakeHostKVCache:
    layout = "page_first_direct"
    layer_num, head_num, head_dim = 2, 2, 64
    dtype = torch.uint8

    def __init__(self, page_size=1, size=1 << 21):
        self.page_size, self.size, self._next = page_size, size, 0
        self.kv_buffer = torch.zeros(
            self.layer_num * size * self.head_num * self.head_dim, dtype=torch.uint8
        )

    def alloc(self, n):
        idx = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return idx

    def free(self, indices):
        return None

    def get_page_buffer_meta(self, indices):
        ptr, base, v_off = [], self.kv_buffer.data_ptr(), (
            self.layer_num * self.size * self.head_num * self.head_dim
            * self.dtype.itemsize
        )
        for i in range(0, len(indices), self.page_size):
            k = base + indices[i] * self.layer_num * self.head_num * self.head_dim * 1
            ptr += [k, k + v_off]
        esz = self.layer_num * self.page_size * self.head_num * self.head_dim
        return ptr, [esz] * len(ptr)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
        ).flatten()

    def set_from_flat_data_page(self, i, d):
        return None


class FakeNode:
    class _K:
        extra_key = None

    def __init__(self):
        self.key = FakeNode._K()

    def protect_host(self):
        pass

    def release_host(self):
        pass


def run(n_tokens=4096, page_size=1, first_check_delay=0.0, policy="best_effort"):
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    mem = FakeHostKVCache(page_size=page_size)
    backend = object.__new__(MooncakeStore)
    backend.store = FakeStore()
    backend.mem_pool_host = mem
    backend.is_mla_backend = False
    backend.mha_suffix, backend.mla_suffix = "0", ""
    backend.extra_backend_tag = None
    backend.prefetch_pgs, backend.prefetch_bandwidth = [], []
    backend.backup_pgs, backend.backup_bandwidth = [], []
    backend.gb_per_page = 0.001

    ctrl = object.__new__(HiCacheController)
    ctrl.page_size, ctrl.storage_batch_size = page_size, 128
    ctrl.prefetch_threshold, ctrl.tp_world_size = 1, 1
    ctrl.enable_storage, ctrl.storage_backend_type = True, "mooncake"
    ctrl.storage_backend = backend
    ctrl.mem_pool_host = mem
    ctrl.get_hash_str = get_hash_str
    ctrl.storage_stop_event = threading.Event()
    for q in ("prefetch_queue", "prefetch_buffer", "prefetch_revoke_queue",
              "ack_backup_queue", "backup_queue", "host_mem_release_queue"):
        setattr(ctrl, q, queue.Queue())
    ctrl.prefetch_tokens_occupied, ctrl.prefetch_capacity_limit = 0, 1 << 40
    ctrl.backup_skip = False
    ctrl.page_get_func = ctrl._page_get_zero_copy
    ctrl.page_set_func = ctrl._page_set_zero_copy

    tree = object.__new__(HiRadixCache)
    tree.page_size, tree.enable_storage, tree.prefetch_threshold = page_size, True, 1
    tree.prefetch_stop_policy = policy
    tree.prefetch_timeout_base, tree.prefetch_timeout_per_page = 1.0, 0.0
    tree.tp_world_size, tree.enable_storage_metrics = 1, False
    tree.hicache_storage_pass_prefix_keys = False
    tree.ongoing_prefetch, tree.prefetch_loaded_tokens_by_reqid = {}, {}
    tree.cache_controller = ctrl
    tree.evict_host = lambda n: None
    tree._insert_helper_host = lambda *a, **k: 0

    marks = {}
    orig_hit, orig_xfer = (
        HiCacheController._storage_hit_query,
        HiCacheController._page_transfer,
    )

    def hit_query(self, operation):
        marks["exists_start"] = time.perf_counter()
        r = orig_hit(self, operation)
        marks["exists_end"] = time.perf_counter()
        return r

    def page_transfer(self, operation):
        marks["xfer_start"] = time.perf_counter()
        r = orig_xfer(self, operation)
        marks["xfer_end"] = time.perf_counter()
        return r

    HiCacheController._storage_hit_query = hit_query
    HiCacheController._page_transfer = page_transfer
    try:
        threading.Thread(
            target=ctrl.prefetch_thread_func, name="prefetch_thread", daemon=True
        ).start()

        t0 = time.perf_counter()
        tree.prefetch_from_storage(
            req_id="r",
            last_host_node=FakeNode(),
            new_input_tokens=list(range(n_tokens)),
            last_hash=None,
            prefix_keys=None,
        )
        op = tree.ongoing_prefetch["r"][3]

        time.sleep(first_check_delay)  # <-- the ONLY variable
        t_check = time.perf_counter()
        done = tree.check_prefetch_progress("r")
        tokens_at_check = op.completed_tokens

        # let background threads drain
        t_end = t_check
        while time.perf_counter() - t_end < 30:
            if "xfer_end" in marks and marks["xfer_end"] > t_check:
                break
            time.sleep(0.002)
        t_end = time.perf_counter()

        ctrl.storage_stop_event.set()
        time.sleep(0.05)
    finally:
        HiCacheController._storage_hit_query = orig_hit
        HiCacheController._page_transfer = orig_xfer

    st = backend.store
    return {
        "delay_ms": first_check_delay * 1000,
        "policy": policy,
        "exists_calls": st.exists_calls,
        "get_calls": st.get_calls,
        "exists_wall_ms": (marks.get("exists_end", 0) - marks.get("exists_start", 0))
        * 1000,
        "xfer_wall_ms": (marks.get("xfer_end", 0) - marks.get("xfer_start", 0)) * 1000,
        "total_ms": (t_end - t0) * 1000,
        "tokens_at_check": tokens_at_check,
        "harvested": done,
    }


def show(r):
    print(
        f"  first_check_delay={r['delay_ms']:7.1f} ms | policy={r['policy']:<13} "
        f"| batch_is_exist={r['exists_calls']:3d}  batch_get_into={r['get_calls']:3d} "
        f"| tokens_delivered={r['tokens_at_check']:5d} "
        f"| exists_wall={r['exists_wall_ms']:7.2f}ms xfer_wall={r['xfer_wall_ms']:6.2f}ms "
        f"| total={r['total_ms']:7.2f}ms"
    )


if __name__ == "__main__":
    print("=" * 118)
    print("4096 tokens, page_size=1.  exists AND get BOTH cost 2 ms/call (identical!).")
    print("Only the scheduler's first check_prefetch_progress() time varies.")
    print("=" * 118)
    for d in (0.0, 0.005, 0.020, 0.080, 0.300):
        show(run(first_check_delay=d, policy="best_effort"))
    print()
    for d in (0.0, 0.080):
        show(run(first_check_delay=d, policy="wait_complete"))
    print("\nDONE")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLA (GLM / DeepSeek-style) HiCache prefetch latency probe for sglang v0.5.9.

MLA specifics that differ from MHA and are modelled here:
  * MooncakeStore.is_mla_backend = True
      -> _get_mla_buffer_meta: ONE key per page (no k/v split)
      -> _batch_postprocess: one bool per page
  * mla_suffix = ""  (pp_size == 1)   -> key = "<hash>__k"
  * page_size = 64 (FlashMLA) or 128 (Cutlass MLA)
  * MLA host pool layout must be page_first / page_first_direct
      -> get_page_buffer_meta: ONE buffer pointer per page

Real sglang code is exercised end to end; only the mooncake C++ object is
replaced by a latency-injecting stand-in.
"""

import queue
import threading
import time

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


class FakeStore:
    def __init__(self, latency=0.002):
        self.latency = latency
        self.lock = threading.Lock()
        self.exists_calls = self.get_calls = 0
        self.exists_keys = self.get_keys = 0
        self.exists_time = self.get_time = 0.0
        self.get_done = threading.Event()

    def batch_is_exist(self, key_strs):
        with self.lock:
            self.exists_calls += 1
            self.exists_keys += len(key_strs)
        t0 = time.perf_counter()
        time.sleep(self.latency)
        with self.lock:
            self.exists_time += time.perf_counter() - t0
        return [1] * len(key_strs)

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        with self.lock:
            self.get_calls += 1
            self.get_keys += len(key_strs)
        t0 = time.perf_counter()
        time.sleep(self.latency)
        with self.lock:
            self.get_time += time.perf_counter() - t0
        self.get_done.set()
        return [max(1, int(s)) for s in buffer_sizes]


class MLAHostKVCache:
    """Mirrors MLATokenToKVPoolHost.get_page_buffer_meta (page_first_direct)."""

    layout = "page_first_direct"
    layer_num = 2
    kv_cache_dim = 128
    dtype = torch.uint8

    def __init__(self, page_size=64, size=1 << 21):
        self.page_size, self.size, self._next = page_size, size, 0
        self.kv_buffer = torch.zeros(
            self.layer_num * size * self.kv_cache_dim, dtype=torch.uint8
        )

    def alloc(self, n):
        idx = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return idx

    def free(self, indices):
        return None

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        ptr_list = []
        base = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        for index in range(0, len(indices), self.page_size):
            ptr_list.append(
                base
                + indices[index]
                * self.layer_num
                * self.kv_cache_dim
                * self.dtype.itemsize
            )
        element_size = (
            self.layer_num * self.dtype.itemsize * self.page_size * self.kv_cache_dim
        )
        return ptr_list, [element_size] * len(ptr_list)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.kv_cache_dim), dtype=self.dtype
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


def build(page_size, latency, n_tp=1):
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    mem = MLAHostKVCache(page_size=page_size)

    backend = object.__new__(MooncakeStore)
    backend.store = FakeStore(latency)
    backend.mem_pool_host = mem
    backend.is_mla_backend = True          # <-- MLA
    backend.mha_suffix = "0"
    backend.mla_suffix = ""                 # <-- pp_size == 1
    backend.extra_backend_tag = None
    backend.prefetch_pgs, backend.prefetch_bandwidth = [], []
    backend.backup_pgs, backend.backup_bandwidth = [], []
    backend.gb_per_page = 0.001

    ctrl = object.__new__(HiCacheController)
    ctrl.page_size, ctrl.storage_batch_size = page_size, 128
    ctrl.prefetch_threshold, ctrl.tp_world_size = 1, n_tp
    ctrl.enable_storage, ctrl.storage_backend_type = True, "mooncake"
    ctrl.storage_backend = backend
    ctrl.mem_pool_host = mem
    ctrl.get_hash_str = get_hash_str
    ctrl.storage_stop_event = threading.Event()
    for q in ("prefetch_queue", "prefetch_buffer", "prefetch_revoke_queue",
              "ack_backup_queue", "backup_queue", "host_mem_release_queue"):
        setattr(ctrl, q, queue.Queue())
    ctrl.prefetch_tokens_occupied, ctrl.prefetch_capacity_limit = 0, 1 << 40
    ctrl.backup_skip = True                 # MLA: only tp_rank 0 backs up
    ctrl.page_get_func = ctrl._page_get_zero_copy
    ctrl.page_set_func = ctrl._page_set_zero_copy

    tree = object.__new__(HiRadixCache)
    tree.page_size, tree.enable_storage, tree.prefetch_threshold = page_size, True, 1
    tree.prefetch_stop_policy = "best_effort"
    tree.prefetch_timeout_base, tree.prefetch_timeout_per_page = 1.0, 0.0
    tree.tp_world_size, tree.enable_storage_metrics = n_tp, False
    tree.hicache_storage_pass_prefix_keys = False
    tree.ongoing_prefetch, tree.prefetch_loaded_tokens_by_reqid = {}, {}
    tree.cache_controller = ctrl
    tree.evict_host = lambda n: None
    tree._insert_helper_host = lambda *a, **k: 0
    return ctrl, backend, tree, mem


# --------------------------------------------------------------------------- #
# 1) RPC-count table: how many round trips does each stage need?
# --------------------------------------------------------------------------- #
def rpc_counts(n_tokens, page_size):
    """Purely arithmetic, mirrors the two chunking loops."""
    exists_chunk_tokens = page_size * 128          # _storage_hit_query
    exists_calls = -(-n_tokens // exists_chunk_tokens)
    hit_pages = -(-n_tokens // page_size)          # assume prefix fully cached
    get_calls = -(-hit_pages // 128)               # _page_transfer
    keys_per_exists = min(page_size * 128, hit_pages)   # MLA: 1 key per page
    keys_per_get = min(128, hit_pages)                  # MLA: 1 key per page
    return exists_calls, keys_per_exists, get_calls, keys_per_get


# --------------------------------------------------------------------------- #
# 2) Decisive experiment: vary WHEN the scheduler gives up
# --------------------------------------------------------------------------- #
def run(n_tokens=4096, page_size=64, latency=0.002, first_check_delay=0.0,
        policy="best_effort", n_reqs=1):
    ctrl, backend, tree, mem = build(page_size, latency)

    m = {"exists_done": threading.Event(), "xfer_done": threading.Event()}
    orig_hit = HiCacheController._storage_hit_query
    orig_xfer = HiCacheController._page_transfer

    def hit_query(self, operation):
        t0 = time.perf_counter()
        r = orig_hit(self, operation)
        m["exists_wall"] = (time.perf_counter() - t0) * 1000
        m["exists_done"].set()
        return r

    def page_transfer(self, operation):
        t0 = time.perf_counter()
        r = orig_xfer(self, operation)
        m["xfer_wall"] = (time.perf_counter() - t0) * 1000
        m["xfer_done"].set()
        return r

    HiCacheController._storage_hit_query = hit_query
    HiCacheController._page_transfer = page_transfer
    try:
        threading.Thread(
            target=ctrl.prefetch_thread_func, name="prefetch_thread", daemon=True
        ).start()

        t_submit = time.perf_counter()
        ops = []
        for i in range(n_reqs):
            tree.prefetch_from_storage(
                req_id=f"r{i}",
                last_host_node=FakeNode(),
                new_input_tokens=list(range(n_tokens)),
                last_hash=None,
                prefix_keys=None,
            )
            ops.append(tree.ongoing_prefetch[f"r{i}"][3])

        time.sleep(first_check_delay)
        t_check = time.perf_counter()
        for i in range(n_reqs):
            tree.check_prefetch_progress(f"r{i}")
        tokens_at_check = [o.completed_tokens for o in ops]

        m["exists_done"].wait(timeout=60)
        m["xfer_done"].wait(timeout=60)
        time.sleep(0.05)
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
        "n_reqs": n_reqs,
        "exists_calls": st.exists_calls,
        "get_calls": st.get_calls,
        "exists_keys": st.exists_keys,
        "get_keys": st.get_keys,
        "exists_wall_ms": m.get("exists_wall", 0.0),
        "xfer_wall_ms": m.get("xfer_wall", 0.0),
        "total_ms": (t_end - t_submit) * 1000,
        "tokens_at_check": tokens_at_check,
    }


def show(r):
    print(
        f"  delay={r['delay_ms']:7.1f}ms reqs={r['n_reqs']} | "
        f"batch_is_exist x{r['exists_calls']:<3d}(keys {r['exists_keys']:<6d}) "
        f"batch_get_into x{r['get_calls']:<3d}(keys {r['get_keys']:<6d}) | "
        f"exists_wall={r['exists_wall_ms']:7.2f}ms xfer_wall={r['xfer_wall_ms']:7.2f}ms "
        f"total={r['total_ms']:7.2f}ms | tokens_delivered={r['tokens_at_check']}"
    )


if __name__ == "__main__":
    print("=" * 118)
    print("MLA (page_size=64, is_mla_backend=True, 1 key/page)")
    print("=" * 118)

    print("\n### 1) RPC count per stage (arithmetic, mirrors the two chunk loops)")
    print(f"  {'tokens':>8} | {'exists calls':>13} {'keys/call':>10} | "
          f"{'get calls':>10} {'keys/call':>10}")
    for n in (2048, 4096, 8192, 16384, 32768, 131072):
        ec, ek, gc, gk = rpc_counts(n, 64)
        print(f"  {n:8d} | {ec:13d} {ek:10d} | {gc:10d} {gk:10d}")

    print("\n### 2) Decisive: vary WHEN the scheduler calls check_prefetch_progress")
    print("    (exists and get cost the SAME 2ms/call -> no latency assumption)")
    for d in (0.0, 0.002, 0.005, 0.020, 0.100):
        show(run(first_check_delay=d))

    print("\n### 3) same, but policy=wait_complete")
    for d in (0.0, 0.020):
        show(run(first_check_delay=d, policy="wait_complete"))

    print("\n### 4) concurrency: 8 requests submitted back-to-back (head-of-line)")
    for n in (1, 4, 8, 16):
        show(run(first_check_delay=0.05, n_reqs=n))

    print("\n### 5) page_size 64 (MLA) vs 1 (MHA-style), 4096 tokens")
    show(run(n_tokens=4096, page_size=64, first_check_delay=0.05))
    show(run(n_tokens=4096, page_size=1, first_check_delay=0.05))

    print("\nDONE")

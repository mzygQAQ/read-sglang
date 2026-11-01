#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify where a HiCache PrefetchOperation's wall-clock time goes, and how much
of it is *wasted* (work that continues after the scheduler already gave up).

Real sglang code (v0.5.9) is exercised end to end:
    HiRadixCache.prefetch_from_storage        (scheduler main thread, submit)
      -> HiCacheController.prefetch             (prefetch_queue.put)
      -> HiCacheController.prefetch_thread_func
           -> _storage_hit_query                -> MooncakeStore.batch_exists
      -> HiCacheController.prefetch_io_aux_func
           -> _page_transfer                    -> MooncakeStore.batch_get_v1
      -> HiRadixCache.check_prefetch_progress   (scheduler main thread, harvest)

The mooncake C++ extension is absent here, so `MooncakeDistributedStore` is
replaced by a stand-in with configurable per-call latency:
    exists_latency : mooncake MASTER metadata round trip
    get_latency    : mooncake SEGMENT data-path round trip
Everything above the C++ boundary is genuine sglang code.
"""

import queue
import sys
import threading
import time

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


class TimingMooncakeStore:
    def __init__(self, exists_latency=0.0, get_latency=0.0):
        self.exists_latency = exists_latency
        self.get_latency = get_latency
        self.exists_calls = self.exists_keys = 0
        self.exists_time = 0.0
        self.get_calls = self.get_keys = 0
        self.get_time = 0.0

    def batch_is_exist(self, key_strs):
        self.exists_calls += 1
        self.exists_keys += len(key_strs)
        t0 = time.perf_counter()
        if self.exists_latency:
            time.sleep(self.exists_latency)
        self.exists_time += time.perf_counter() - t0
        return [1] * len(key_strs)

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        self.get_calls += 1
        self.get_keys += len(key_strs)
        t0 = time.perf_counter()
        if self.get_latency:
            time.sleep(self.get_latency)
        self.get_time += time.perf_counter() - t0
        return [max(1, int(s)) for s in buffer_sizes]


class FakeHostKVCache:
    layout = "page_first_direct"
    layer_num = 2
    head_num = 2
    head_dim = 64
    dtype = torch.uint8

    def __init__(self, size=1 << 21, page_size=1):
        self.page_size = page_size
        self.size = size
        self._next = 0
        self.kv_buffer = torch.zeros(
            self.layer_num * size * self.head_num * self.head_dim, dtype=torch.uint8
        )

    def alloc(self, need_size):
        idx = torch.arange(self._next, self._next + need_size, dtype=torch.int64)
        self._next += need_size
        return idx

    def free(self, indices):
        return None

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        ptr_list, base, indices = [], self.kv_buffer.data_ptr(), indices.tolist()
        v_off = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            k = (
                base
                + indices[index]
                * self.layer_num
                * self.head_num
                * self.head_dim
                * self.dtype.itemsize
            )
            ptr_list.append(k)
            ptr_list.append(k + v_off)
        esz = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        return ptr_list, [esz] * len(ptr_list)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        return None


class FakeNode:
    class _Key:
        extra_key = None

    def __init__(self):
        self.key = FakeNode._Key()

    def protect_host(self):
        pass

    def release_host(self):
        pass


def build(page_size, exists_latency, get_latency):
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    mem_pool_host = FakeHostKVCache(page_size=page_size)

    backend = object.__new__(MooncakeStore)
    backend.store = TimingMooncakeStore(exists_latency, get_latency)
    backend.mem_pool_host = mem_pool_host
    backend.is_mla_backend = False
    backend.mha_suffix = "0"
    backend.mla_suffix = ""
    backend.extra_backend_tag = None
    backend.prefetch_pgs = []
    backend.prefetch_bandwidth = []
    backend.backup_pgs = []
    backend.backup_bandwidth = []
    backend.gb_per_page = 0.001

    ctrl = object.__new__(HiCacheController)
    ctrl.page_size = page_size
    ctrl.storage_batch_size = 128
    ctrl.prefetch_threshold = 1
    ctrl.tp_world_size = 1
    ctrl.enable_storage = True
    ctrl.storage_backend = backend
    ctrl.storage_backend_type = "mooncake"
    ctrl.mem_pool_host = mem_pool_host
    ctrl.get_hash_str = get_hash_str
    ctrl.storage_stop_event = threading.Event()
    ctrl.prefetch_queue = queue.Queue()
    ctrl.prefetch_buffer = queue.Queue()
    ctrl.prefetch_revoke_queue = queue.Queue()
    ctrl.ack_backup_queue = queue.Queue()
    ctrl.backup_queue = queue.Queue()
    ctrl.host_mem_release_queue = queue.Queue()
    ctrl.prefetch_tokens_occupied = 0
    ctrl.prefetch_capacity_limit = 1 << 40
    ctrl.backup_skip = False
    ctrl.page_get_func = ctrl._page_get_zero_copy
    ctrl.page_set_func = ctrl._page_set_zero_copy

    tree = object.__new__(HiRadixCache)
    tree.page_size = page_size
    tree.enable_storage = True
    tree.prefetch_threshold = 1
    tree.prefetch_stop_policy = "best_effort"
    tree.prefetch_timeout_base = 1.0
    tree.prefetch_timeout_per_page = 0.0
    tree.tp_world_size = 1
    tree.enable_storage_metrics = False
    tree.hicache_storage_pass_prefix_keys = False
    tree.ongoing_prefetch = {}
    tree.prefetch_loaded_tokens_by_reqid = {}
    tree.cache_controller = ctrl
    tree.evict_host = lambda n: None
    tree._insert_helper_host = lambda node, key, host_value, hash_value: 0
    return ctrl, backend, tree


def run(n_tokens, page_size, exists_latency, get_latency, policy, scheduler_tick):
    ctrl, backend, tree = build(page_size, exists_latency, get_latency)
    tree.prefetch_stop_policy = policy

    m = {}
    orig_hit = HiCacheController._storage_hit_query
    orig_xfer = HiCacheController._page_transfer
    orig_prefetch = HiCacheController.prefetch

    def hit_query(self, operation):
        m["exists_start"] = time.perf_counter()
        r = orig_hit(self, operation)
        m["exists_end"] = time.perf_counter()
        m["exists_done"].set()
        return r

    def page_transfer(self, operation):
        m["xfer_start"] = time.perf_counter()
        r = orig_xfer(self, operation)
        m["xfer_end"] = time.perf_counter()
        m["xfer_done"].set()
        return r

    def prefetch(self, *a, **kw):
        m["submit"] = time.perf_counter()
        return orig_prefetch(self, *a, **kw)

    m["exists_done"] = threading.Event()
    m["xfer_done"] = threading.Event()
    HiCacheController._storage_hit_query = hit_query
    HiCacheController._page_transfer = page_transfer
    HiCacheController.prefetch = prefetch

    try:
        threading.Thread(
            target=ctrl.prefetch_thread_func, name="prefetch_thread", daemon=True
        ).start()

        tree.prefetch_from_storage(
            req_id="req-1",
            last_host_node=FakeNode(),
            new_input_tokens=list(range(n_tokens)),
            last_hash=None,
            prefix_keys=None,
        )
        op = tree.ongoing_prefetch["req-1"][3]
        t_submit = m["submit"]

        # ---- scheduler main thread: harvest loop ----------------------- #
        t_harvest, done = None, False
        deadline = time.perf_counter() + 120
        while time.perf_counter() < deadline:
            if tree.check_prefetch_progress("req-1"):
                t_harvest = time.perf_counter()
                done = True
                break
            time.sleep(scheduler_tick)

        tokens_at_harvest = op.completed_tokens

        # ---- let the background threads finish their (possibly wasted) work #
        m["exists_done"].wait(timeout=120)
        m["xfer_done"].wait(timeout=120)
        time.sleep(0.05)
        t_all_done = time.perf_counter()

        ctrl.storage_stop_event.set()
        time.sleep(0.05)
    finally:
        HiCacheController._storage_hit_query = orig_hit
        HiCacheController._page_transfer = orig_xfer
        HiCacheController.prefetch = orig_prefetch

    st = backend.store
    return {
        "n_tokens": n_tokens,
        "page_size": page_size,
        "policy": policy,
        "harvest_ms": (t_harvest - t_submit) * 1000 if t_harvest else float("nan"),
        "full_ms": (t_all_done - t_submit) * 1000,
        "waste_ms": (t_all_done - t_harvest) * 1000 if t_harvest else float("nan"),
        "q1_ms": (m.get("exists_start", t_submit) - t_submit) * 1000,
        "exists_ms": (
            m["exists_end"] - m["exists_start"]
        )
        * 1000
        if "exists_end" in m
        else float("nan"),
        "q2_ms": (m.get("xfer_start", m.get("exists_end", 0)) - m.get("exists_end", 0))
        * 1000,
        "xfer_ms": (m["xfer_end"] - m["xfer_start"]) * 1000
        if "xfer_end" in m
        else 0.0,
        "exists_calls": st.exists_calls,
        "exists_keys": st.exists_keys,
        "get_calls": st.get_calls,
        "get_keys": st.get_keys,
        "fake_exists_ms": st.exists_time * 1000,
        "fake_get_ms": st.get_time * 1000,
        "tokens_at_harvest": tokens_at_harvest,
        "tokens_total": op.completed_tokens,
        "revoked": ctrl.prefetch_revoke_queue.qsize() > 0,
        "harvested": done,
    }


def show(r):
    print(
        f"\n--- {r['n_tokens']} tokens | page_size={r['page_size']} | "
        f"policy={r['policy']} ---"
    )
    print(f"  harvest (scheduler gave up)      {r['harvest_ms']:8.2f} ms   "
          f"tokens delivered = {r['tokens_at_harvest']}")
    print(f"  op fully finished (bg threads)   {r['full_ms']:8.2f} ms")
    print(f"  >>> WASTED after harvest         {r['waste_ms']:8.2f} ms")
    print(f"    [1] wait prefetch_queue        {r['q1_ms']:8.2f} ms")
    print(f"    [2] _storage_hit_query         {r['exists_ms']:8.2f} ms  "
          f"calls={r['exists_calls']} keys={r['exists_keys']} "
          f"(fake RPC {r['fake_exists_ms']:.2f} ms)")
    print(f"    [3] wait prefetch_buffer       {r['q2_ms']:8.2f} ms")
    print(f"    [4] _page_transfer             {r['xfer_ms']:8.2f} ms  "
          f"calls={r['get_calls']} keys={r['get_keys']} "
          f"(fake RPC {r['fake_get_ms']:.2f} ms)")
    print(f"  revoked={r['revoked']}  tokens_at_harvest={r['tokens_at_harvest']}  "
          f"tokens_total={r['tokens_total']}")


if __name__ == "__main__":
    E, G = 0.002, 0.0005  # 2ms master RTT, 0.5ms segment RTT

    print("=" * 80)
    print("A) best_effort (DEFAULT) vs wait_complete, page_size=1, 4096 tokens")
    print("=" * 80)
    show(run(4096, 1, E, G, "best_effort", 0.0005))
    show(run(4096, 1, E, G, "wait_complete", 0.0005))

    print("\n" + "=" * 80)
    print("B) best_effort, ZERO network latency -> pure Python/structural cost")
    print("=" * 80)
    for n in (4096, 16384, 65536):
        show(run(n, 1, 0.0, 0.0, "wait_complete", 0.0005))

    print("\n" + "=" * 80)
    print("C) page_size=64 (MLA/FA3 style) vs page_size=1 (MHA default), 4096 tok")
    print("=" * 80)
    show(run(4096, 64, E, G, "wait_complete", 0.0005))

    print("\nDONE")

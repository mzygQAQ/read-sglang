#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Where does a PrefetchOperation's wall-clock time actually go, under load?

Established so far (real sglang v0.5.9 + real Mooncake source):
  * MLA: batch_is_exist and batch_get_into are called 1:1 (same chunking)
  * both hit the mooncake MASTER (BatchExistKey vs BatchQuery)
  * batch_get_into does strictly MORE work per call than batch_is_exist
  => per-call latency of exists <= get.  So "total >> page_transfer" can NOT be
     explained by the exists stage being slower.

Remaining candidate: WAITING.  The prefetch pipeline is TWO serial stages with
ONE thread each:
    prefetch_queue  --[prefetch_thread]-->  prefetch_buffer
                       --[prefetch_io_aux_thread]-->  done
and the io thread finishes one operation's whole _page_transfer before taking
the next one.

This harness measures the queueing delay directly.
"""

import queue
import threading
import time

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


class FakeStore:
    """Latency-injecting mooncake stand-in. Sleeps are NOT globally serialized,
    so concurrent calls really do overlap (like real mooncake)."""

    def __init__(self, exists_latency=0.002, get_latency=0.002):
        self.exists_latency = exists_latency
        self.get_latency = get_latency
        self.lock = threading.Lock()
        self.exists_calls = self.get_calls = 0
        self.get_time = self.exists_time = 0.0

    def batch_is_exist(self, key_strs):
        with self.lock:
            self.exists_calls += 1
        t0 = time.perf_counter()
        time.sleep(self.exists_latency)
        with self.lock:
            self.exists_time += time.perf_counter() - t0
        return [1] * len(key_strs)

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        with self.lock:
            self.get_calls += 1
        t0 = time.perf_counter()
        time.sleep(self.get_latency)
        with self.lock:
            self.get_time += time.perf_counter() - t0
        return [max(1, int(s)) for s in buffer_sizes]


class MLAHostKVCache:
    layout = "page_first_direct"
    layer_num, kv_cache_dim = 2, 128
    dtype = torch.uint8

    def __init__(self, page_size=64, size=1 << 22):
        self.page_size, self.size, self._next = page_size, size, 0
        self.kv_buffer = torch.zeros(
            self.layer_num * size * self.kv_cache_dim, dtype=torch.uint8
        )

    def alloc(self, n):
        idx = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return idx

    def free(self, i):
        return None

    def get_page_buffer_meta(self, indices):
        ptr, base = [], self.kv_buffer.data_ptr()
        for i in range(0, len(indices), self.page_size):
            ptr.append(base + indices[i] * self.layer_num * self.kv_cache_dim)
        esz = self.layer_num * self.page_size * self.kv_cache_dim
        return ptr, [esz] * len(ptr)

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


def build(page_size, exists_latency, get_latency):
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    mem = MLAHostKVCache(page_size=page_size)
    backend = object.__new__(MooncakeStore)
    backend.store = FakeStore(exists_latency, get_latency)
    backend.mem_pool_host = mem
    backend.is_mla_backend = True
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
    ctrl.backup_skip = True
    ctrl.page_get_func = ctrl._page_get_zero_copy
    ctrl.page_set_func = ctrl._page_set_zero_copy

    tree = object.__new__(HiRadixCache)
    tree.page_size, tree.enable_storage, tree.prefetch_threshold = page_size, True, 1
    tree.prefetch_stop_policy = "best_effort"
    tree.prefetch_timeout_base, tree.prefetch_timeout_per_page = 1.0, 0.0
    tree.tp_world_size, tree.enable_storage_metrics = 1, False
    tree.hicache_storage_pass_prefix_keys = False
    tree.ongoing_prefetch, tree.prefetch_loaded_tokens_by_reqid = {}, {}
    tree.cache_controller = ctrl
    tree.evict_host = lambda n: None
    tree._insert_helper_host = lambda *a, **k: 0
    return ctrl, backend, tree


def run(n_reqs, n_tokens=8192, page_size=64,
        exists_latency=0.002, get_latency=0.002):
    ctrl, backend, tree = build(page_size, exists_latency, get_latency)

    # per-op timing of the transfer stage
    starts, ends = {}, {}
    orig_xfer = HiCacheController._page_transfer

    def page_transfer(self, operation):
        rid = operation.request_id
        starts[rid] = time.perf_counter()
        r = orig_xfer(self, operation)
        ends[rid] = time.perf_counter()
        return r

    HiCacheController._page_transfer = page_transfer
    try:
        threading.Thread(
            target=ctrl.prefetch_thread_func, name="prefetch_thread", daemon=True
        ).start()

        t0 = time.perf_counter()
        for i in range(n_reqs):
            tree.prefetch_from_storage(
                req_id=f"r{i}",
                last_host_node=FakeNode(),
                new_input_tokens=list(range(n_tokens)),
                last_hash=None,
                prefix_keys=None,
            )
        # wait for every transfer to finish
        deadline = time.perf_counter() + 300
        while len(ends) < n_reqs and time.perf_counter() < deadline:
            time.sleep(0.002)
        t_all = time.perf_counter()

        ctrl.storage_stop_event.set()
        time.sleep(0.05)
    finally:
        HiCacheController._page_transfer = orig_xfer

    st = backend.store
    waits = [(starts[f"r{i}"] - t0) * 1000 for i in range(n_reqs)]
    xfers = [(ends[f"r{i}"] - starts[f"r{i}"]) * 1000 for i in range(n_reqs)]
    return {
        "n_reqs": n_reqs,
        "n_tokens": n_tokens,
        "total_ms": (t_all - t0) * 1000,
        "exists_calls": st.exists_calls,
        "get_calls": st.get_calls,
        "exists_fake_ms": st.exists_time * 1000,
        "get_fake_ms": st.get_time * 1000,
        "wait_first": waits[0],
        "wait_last": waits[-1],
        "xfer_first": xfers[0],
        "xfer_last": xfers[-1],
        "xfer_avg": sum(xfers) / len(xfers),
    }


def show(r):
    print(
        f"  reqs={r['n_reqs']:3d} tokens={r['n_tokens']:6d} | "
        f"exists_calls={r['exists_calls']:3d} get_calls={r['get_calls']:3d} | "
        f"全部完成耗时={r['total_ms']:8.1f}ms | "
        f"排队等待 首={r['wait_first']:7.1f}ms 末={r['wait_last']:7.1f}ms | "
        f"_page_transfer 自身 首={r['xfer_first']:5.2f}ms 末={r['xfer_last']:5.2f}ms "
        f"均值={r['xfer_avg']:5.2f}ms"
    )


if __name__ == "__main__":
    print("=" * 132)
    print("MLA page_size=64, 8192 tokens/req.  mooncake: exists=2ms/call, get=2ms/call")
    print("=" * 132)

    print("\n### A) 并发压力：一次性提交 N 个请求，观察排队")
    for n in (1, 4, 16, 64, 256):
        show(run(n))

    print("\n### B) 同样负载，但砍掉 exists 阶段（模拟合并 exists+get）")
    for n in (16, 64, 256):
        show(run(n, exists_latency=0.0))

    print("\n### C) 流水线吞吐对比 (N=256)")
    a = run(256)
    b = run(256, exists_latency=0.0)
    print(f"  有 exists 阶段: 256 req 全部完成 {a['total_ms']:.1f}ms "
          f"→ {256 / (a['total_ms'] / 1000):.1f} req/s")
    print(f"  无 exists 阶段: 256 req 全部完成 {b['total_ms']:.1f}ms "
          f"→ {256 / (b['total_ms'] / 1000):.1f} req/s")

    print("\n### D) 延迟敏感性扫描 (N=256)：绝对值由【假设的 RPC 延迟】决定")
    print("  lat=0 时剩余的全部是【模型无关】的 Python/结构开销")
    print(f"  {'假设 exists=lat,get=lat':<26} {'全部完成':>10} {'每op串行':>11} "
          f"{'排队等待(末)':>13} {'_page_transfer':>14}")
    for lat in (0.0, 0.0002, 0.0005, 0.002, 0.005, 0.02):
        r = run(256, exists_latency=lat, get_latency=lat)
        print(f"  lat={lat*1000:>7.2f}ms{'':<12} {r['total_ms']:>9.1f}ms "
              f"{r['total_ms']/256:>10.3f}ms {r['wait_last']:>12.1f}ms "
              f"{r['xfer_avg']:>13.2f}ms")

    print("\nDONE")

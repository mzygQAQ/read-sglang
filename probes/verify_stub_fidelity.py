#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
替身保真度自检 (stub fidelity verification)
============================================

本仓库的分析结论来自「用替身替换 mooncake C++ 扩展后，驱动 **真实 sglang 代码路径**」的测试。
这个脚本把「替身是否忠实」这件事从「相信我」变成「可验证」：

  V1  host 内存池 `get_page_buffer_meta` 替身  vs  真实 `MLATokenToKVPoolHost.get_page_buffer_meta`
      → 逐个指针地址、逐个 element_size 精确比对
  V2  `_storage_hit_query` 的分片算术       vs  真实方法实际发出的 `batch_exists` 次数 / 每次 key 数
  V3  `_page_transfer` 的批次数             vs  真实方法实际发出的 `batch_get_v1` 次数
  V4  线程拓扑                             →  断言 `batch_get_into` 确实跑在 `prefetch_io_aux_thread` 上
  V5  收割时机语义                         →  断言 best_effort 下「收割早于 IO 完成 ⇒ 交付 0 token」

跑法：
    wsl -d Ubuntu-24.04 -- bash -lc \
      "cd /home/mzygd/workspace/sglang && .venv/bin/python <本脚本路径>"

退出码 0 = 全部通过。
"""

import queue
import sys
import threading
import time

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# 替身组件
# --------------------------------------------------------------------------- #
class StubHostKVCache:
    """`MLATokenToKVPoolHost` 的最小替身。

    `get_page_buffer_meta` 是**逐行照抄**自 memory_pool_host.py:987 的 page_first_direct 分支。
    V1 会用真实类的方法做等价性校验。
    """

    layout = "page_first_direct"
    layer_num = 2
    kv_cache_dim = 128
    dtype = torch.uint8

    def __init__(self, page_size=64, size=4096):
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

    # ↓↓↓ 逐行照抄 memory_pool_host.py:987 (page_first_direct 分支) ↓↓↓
    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        for index in range(0, len(indices), self.page_size):
            k_ptr = (
                kv_buffer_data_ptr
                + indices[index]
                * self.layer_num
                * self.kv_cache_dim
                * self.dtype.itemsize
            )
            ptr_list.append(k_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.kv_cache_dim
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list
    # ↑↑↑ 照抄结束 ↑↑↑

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.kv_cache_dim), dtype=self.dtype
        ).flatten()

    def set_from_flat_data_page(self, i, d):
        return None


class RecordingStore:
    """mooncake `MooncakeDistributedStore` 的替身：记录调用，不真的打网络。"""

    def __init__(self, exists_latency=0.0, get_latency=0.0):
        self.exists_latency, self.get_latency = exists_latency, get_latency
        self.exists_calls, self.exists_key_counts = 0, []
        self.get_calls, self.get_key_counts = 0, []
        self.get_threads = []

    def batch_is_exist(self, key_strs):
        self.exists_calls += 1
        self.exists_key_counts.append(len(key_strs))
        if self.exists_latency:
            time.sleep(self.exists_latency)
        return [1] * len(key_strs)

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        self.get_calls += 1
        self.get_key_counts.append(len(key_strs))
        self.get_threads.append(threading.current_thread().name)
        if self.get_latency:
            time.sleep(self.get_latency)
        return [max(1, int(s)) for s in buffer_sizes]


def make_backend(mem_pool_host, store):
    """用真实 MooncakeStore 类，但把 C++ 对象换掉。"""
    b = object.__new__(MooncakeStore)
    b.store = store
    b.mem_pool_host = mem_pool_host
    b.is_mla_backend = True          # MLA
    b.mha_suffix, b.mla_suffix = "0", ""
    b.extra_backend_tag = None
    b.prefetch_pgs, b.prefetch_bandwidth = [], []
    b.backup_pgs, b.backup_bandwidth = [], []
    b.gb_per_page = 0.001
    return b


def make_controller(backend, mem_pool_host, page_size):
    ctrl = object.__new__(HiCacheController)
    ctrl.page_size, ctrl.storage_batch_size = page_size, 128
    ctrl.prefetch_threshold, ctrl.tp_world_size = 1, 1
    ctrl.enable_storage, ctrl.storage_backend_type = True, "mooncake"
    ctrl.storage_backend = backend
    ctrl.mem_pool_host = mem_pool_host
    ctrl.get_hash_str = get_hash_str
    ctrl.storage_stop_event = threading.Event()
    for q in ("prefetch_queue", "prefetch_buffer", "prefetch_revoke_queue",
              "ack_backup_queue", "backup_queue", "host_mem_release_queue"):
        setattr(ctrl, q, queue.Queue())
    ctrl.prefetch_tokens_occupied, ctrl.prefetch_capacity_limit = 0, 1 << 40
    ctrl.backup_skip = True
    ctrl.page_get_func = ctrl._page_get_zero_copy
    ctrl.page_set_func = ctrl._page_set_zero_copy
    return ctrl


# =========================================================================== #
# V1  get_page_buffer_meta 替身 vs 真实实现
# =========================================================================== #
def v1_meta_equivalence():
    print("\nV1  host 内存池 get_page_buffer_meta 替身 vs 真实 MLATokenToKVPoolHost")
    mem = StubHostKVCache(page_size=64)
    indices = mem.alloc(64 * 5)  # 5 页

    real_ptrs, real_sizes = MLATokenToKVPoolHost.get_page_buffer_meta(mem, indices)
    stub_ptrs, stub_sizes = mem.get_page_buffer_meta(indices)

    check("指针个数一致", len(real_ptrs) == len(stub_ptrs),
          f"real={len(real_ptrs)} stub={len(stub_ptrs)}")
    check("每个指针地址逐一相等", real_ptrs == stub_ptrs)
    check("element_size 列表相等", real_sizes == stub_sizes,
          f"real={real_sizes[:1]} stub={stub_sizes[:1]}")

    # MLA 每页 1 个 ptr（不拆 k/v）
    check("MLA 每页恰好 1 个 buffer 指针",
          len(real_ptrs) == len(indices) // mem.page_size,
          f"{len(real_ptrs)} ptrs for {len(indices)} indices (page_size={mem.page_size})")


# =========================================================================== #
# V2 / V3  分片算术 vs 真实方法实际发出的调用
# =========================================================================== #
def v2v3_chunking(page_size=64, n_tokens=8192):
    print(f"\nV2/V3  分片算术 vs 真实方法实际发出的调用  (page_size={page_size}, tokens={n_tokens})")
    mem = StubHostKVCache(page_size=page_size)
    store = RecordingStore()
    backend = make_backend(mem, store)
    ctrl = make_controller(backend, mem, page_size)

    # ---- V2: 真实 _storage_hit_query ---- #
    op = ctrl.prefetch_queue.get_nowait if False else None
    from sglang.srt.managers.cache_controller import PrefetchOperation

    operation = PrefetchOperation(
        request_id="v2",
        host_indices=mem.alloc(n_tokens),
        token_ids=list(range(n_tokens)),
        last_hash=None,
        prefix_keys=None,
    )
    hash_value, hit_count = ctrl._storage_hit_query(operation)

    # 算术公式：exists 分片 = page_size * storage_batch_size token/次
    exists_chunk_tokens = page_size * ctrl.storage_batch_size
    expect_exists_calls = -(-n_tokens // exists_chunk_tokens)
    expect_exists_keys = min(exists_chunk_tokens // page_size, len(hash_value))

    check("真实 `_storage_hit_query` 发出的 batch_exists 次数 == 算术公式",
          store.exists_calls == expect_exists_calls,
          f"real={store.exists_calls} formula={expect_exists_calls}")
    check("每次 batch_exists 的 key 数 == 分片内的页数",
          all(c == expect_exists_keys for c in store.exists_key_counts),
          f"real={store.exists_key_counts} expect_each={expect_exists_keys}")
    check("命中页数 == tokens/page_size",
          len(hash_value) == n_tokens // page_size,
          f"{len(hash_value)} pages for {n_tokens} tokens")

    # ---- V3: 真实 _page_transfer ---- #
    operation.hash_value = hash_value
    operation.host_indices = operation.host_indices[: len(hash_value) * page_size]
    store.get_calls = 0
    store.get_key_counts = []
    ctrl._page_transfer(operation)

    expect_get_calls = -(-len(hash_value) // ctrl.storage_batch_size)
    expect_get_keys = min(ctrl.storage_batch_size, len(hash_value))
    check("真实 `_page_transfer` 发出的 batch_get_v1 次数 == 算术公式",
          store.get_calls == expect_get_calls,
          f"real={store.get_calls} formula={expect_get_calls}")
    check("每次 batch_get_v1 的 key 数 == 页数(MLA 每页 1 key)",
          all(c == expect_get_keys for c in store.get_key_counts),
          f"real={store.get_key_counts} expect_each={expect_get_keys}")


# =========================================================================== #
# V4  线程拓扑：batch_get_into 必须跑在 prefetch_io_aux_thread 上
# =========================================================================== #
def v4_thread_topology():
    print("\nV4  线程拓扑：batch_get_into 的调用线程")
    mem = StubHostKVCache(page_size=64)
    store = RecordingStore()
    backend = make_backend(mem, store)
    ctrl = make_controller(backend, mem, 64)

    from sglang.srt.managers.cache_controller import PrefetchOperation

    op = PrefetchOperation(
        request_id="v4",
        host_indices=mem.alloc(64 * 4),
        token_ids=list(range(64 * 4)),
        last_hash=None,
        prefix_keys=None,
    )
    op.hash_value = [f"h{i}" for i in range(4)]
    ctrl.prefetch_buffer.put(op)

    t = threading.Thread(target=ctrl.prefetch_io_aux_func,
                         name="prefetch_io_aux", daemon=True)
    t.start()
    deadline = time.time() + 10
    while not store.get_threads and time.time() < deadline:
        time.sleep(0.005)
    ctrl.storage_stop_event.set()
    time.sleep(0.05)

    check("batch_get_into 确实被调用了", len(store.get_threads) > 0,
          f"calls={len(store.get_threads)}")
    check("调用线程不是主线程",
          len(store.get_threads) > 0 and all(n != "MainThread" for n in store.get_threads),
          f"threads={store.get_threads}")
    check("调用线程是 prefetch_io_aux_thread",
          len(store.get_threads) > 0 and all(n == "prefetch_io_aux" for n in store.get_threads),
          f"threads={store.get_threads}")


# =========================================================================== #
# V5  收割时机语义：best_effort 早收割 ⇒ 交付 0 token
# =========================================================================== #
def v5_harvest_timing():
    print("\nV5  收割时机语义 (best_effort)")
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    def run(delay):
        mem = StubHostKVCache(page_size=64)
        store = RecordingStore(exists_latency=0.002, get_latency=0.002)
        backend = make_backend(mem, store)
        ctrl = make_controller(backend, mem, 64)

        tree = object.__new__(HiRadixCache)
        tree.page_size, tree.enable_storage, tree.prefetch_threshold = 64, True, 1
        tree.prefetch_stop_policy = "best_effort"
        tree.prefetch_timeout_base, tree.prefetch_timeout_per_page = 1.0, 0.0
        tree.tp_world_size, tree.enable_storage_metrics = 1, False
        tree.hicache_storage_pass_prefix_keys = False
        tree.ongoing_prefetch, tree.prefetch_loaded_tokens_by_reqid = {}, {}
        tree.cache_controller = ctrl
        tree.evict_host = lambda n: None
        tree._insert_helper_host = lambda *a, **k: 0

        class N:
            class _K:
                extra_key = None

            def __init__(self):
                self.key = N._K()

            def protect_host(self):
                pass

            def release_host(self):
                pass

        threading.Thread(target=ctrl.prefetch_thread_func,
                         name="prefetch_thread", daemon=True).start()
        tree.prefetch_from_storage(
            req_id="v5", last_host_node=N(),
            new_input_tokens=list(range(64 * 4)), last_hash=None, prefix_keys=None,
        )
        operation = tree.ongoing_prefetch["v5"][3]
        time.sleep(delay)
        tree.check_prefetch_progress("v5")
        tokens = operation.completed_tokens
        ctrl.storage_stop_event.set()
        time.sleep(0.05)
        return tokens

    early = run(0.0)
    late = run(0.2)
    check("收割早于 IO 完成 ⇒ 交付 0 token", early == 0, f"completed_tokens={early}")
    check("收割晚于 IO 完成 ⇒ 交付全部 token",
          late == 64 * 4, f"completed_tokens={late} (expect {64*4})")


if __name__ == "__main__":
    print("=" * 78)
    print("替身保真度自检 —— 验证本仓库结论所依赖的测试替身是否忠实")
    print("=" * 78)
    v1_meta_equivalence()
    v2v3_chunking(page_size=64, n_tokens=8192)
    v2v3_chunking(page_size=1, n_tokens=4096)   # MHA 默认 page_size
    v4_thread_topology()
    v5_harvest_timing()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} 项未通过 → {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED —— 替身与真实 sglang 实现一致，结论的「结构部分」成立")
    print("=" * 78)

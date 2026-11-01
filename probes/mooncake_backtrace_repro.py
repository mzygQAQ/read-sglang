#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump the REAL Python call stack (faulthandler backtrace format) of the thread(s)
that call Mooncake's batch_get_xx APIs inside sglang v0.5.9.

Real sglang code exercised (repo is NOT modified; the only runtime patch is the
instrumentation wrapper around HiCacheController.prefetch, which delegates to
the original implementation):

  python/sglang/srt/managers/cache_controller.py
      HiCacheController.prefetch_io_aux_func   -> thread "prefetch_io_aux"
      HiCacheController.prefetch_thread_func    -> thread "prefetch_thread"
      HiCacheController._page_transfer
      HiCacheController._page_get_zero_copy
      HiCacheController._generic_page_get
      HiCacheController._storage_hit_query
  python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py
      MooncakeStore.batch_get_v1 / batch_exists
  python/sglang/srt/mem_cache/hiradix_cache.py
      HiRadixCache.prefetch_from_storage        -> scheduler main thread

The mooncake C++ pybind extension is not installed in this venv, so the
`MooncakeDistributedStore` object (`self.store`) is replaced by a stand-in that
emits a faulthandler backtrace of the calling thread and then returns plausible
results. Everything above that boundary is the genuine sglang code path.
"""

import faulthandler
import queue
import sys
import threading
import time

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

PAGE_SIZE = 4
N_PAGES = 8


def dump(tag):
    print("=" * 100)
    print(f"### {tag}")
    print("=" * 100)
    sys.stdout.flush()
    faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Stand-in for mooncake.store.MooncakeDistributedStore (the C++ pybind object)
# --------------------------------------------------------------------------- #
class FakeMooncakePybindStore:
    def __init__(self):
        self.get_into_calls = 0
        self.is_exist_calls = 0
        self.done = threading.Event()

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        self.get_into_calls += 1
        dump(f"MOONCAKE C++  batch_get_into()  call#{self.get_into_calls}  "
             f"keys={len(key_strs)}")
        self.done.set()
        # mooncake returns the number of bytes read per key (negative = error)
        return [max(1, int(sz)) for sz in buffer_sizes]

    def batch_is_exist(self, key_strs):
        self.is_exist_calls += 1
        dump(f"MOONCAKE C++  batch_is_exist()  call#{self.is_exist_calls}  "
             f"keys={len(key_strs)}")
        return [1] * len(key_strs)


# --------------------------------------------------------------------------- #
# Minimal stand-in for HostKVCache (page_first_direct layout, as mooncake wants)
# --------------------------------------------------------------------------- #
class FakeHostKVCache:
    layout = "page_first_direct"
    layer_num = 2
    head_num = 2
    head_dim = 64
    dtype = torch.uint8

    def __init__(self, size=4096):
        self.page_size = PAGE_SIZE
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
        """Faithful copy of memory_pool_host.py:623 (page_first_direct branch)."""
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            k_ptr = (
                kv_buffer_data_ptr
                + indices[index]
                * self.layer_num
                * self.head_num
                * self.head_dim
                * self.dtype.itemsize
            )
            v_ptr = k_ptr + v_offset
            ptr_list.append(k_ptr)
            ptr_list.append(v_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        return ptr_list, [element_size] * len(ptr_list)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        return None


# --------------------------------------------------------------------------- #
# Build a real MooncakeStore without touching the mooncake C++ extension
# --------------------------------------------------------------------------- #
def make_mooncake_store(mem_pool_host):
    store = object.__new__(MooncakeStore)
    store.store = FakeMooncakePybindStore()
    store.mem_pool_host = mem_pool_host
    store.is_mla_backend = False
    store.mha_suffix = "0"
    store.mla_suffix = ""
    store.extra_backend_tag = None
    store.prefetch_pgs = []
    store.prefetch_bandwidth = []
    store.backup_pgs = []
    store.backup_bandwidth = []
    store.gb_per_page = 0.001
    return store


# --------------------------------------------------------------------------- #
# Build a stub `self` for HiCacheController, keeping the REAL methods
# --------------------------------------------------------------------------- #
def make_controller(storage_backend, mem_pool_host):
    ctrl = object.__new__(HiCacheController)
    ctrl.page_size = PAGE_SIZE
    ctrl.storage_batch_size = 128
    ctrl.prefetch_threshold = 1
    ctrl.tp_world_size = 1
    ctrl.enable_storage = True
    ctrl.storage_backend = storage_backend
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
    ctrl.prefetch_capacity_limit = 1 << 30
    # mooncake always uses the zero-copy v1 interface
    ctrl.page_get_func = ctrl._page_get_zero_copy
    ctrl.page_set_func = ctrl._page_set_zero_copy
    return ctrl


# --------------------------------------------------------------------------- #
# Scenario A: the "prefetch_io_aux" background thread -> mooncake batch_get_v1
# --------------------------------------------------------------------------- #
def scenario_a():
    print("\n\n" + "#" * 100)
    print("# SCENARIO A : thread that calls mooncake batch_get_v1 -> batch_get_into")
    print("#" * 100 + "\n")
    mem_pool_host = FakeHostKVCache()
    backend = make_mooncake_store(mem_pool_host)
    ctrl = make_controller(backend, mem_pool_host)

    op = PrefetchOperation(
        request_id="req-A",
        host_indices=mem_pool_host.alloc(N_PAGES * PAGE_SIZE),
        token_ids=list(range(N_PAGES * PAGE_SIZE)),
        last_hash=None,
        prefix_keys=None,
    )
    op.hash_value = [f"hash_{i}" for i in range(N_PAGES)]

    ctrl.prefetch_buffer.put(op)

    t = threading.Thread(
        target=HiCacheController.prefetch_io_aux_func,
        args=(ctrl,),
        name="prefetch_io_aux",
        daemon=True,
    )
    t.start()
    backend.store.done.wait(timeout=10)
    time.sleep(0.3)
    ctrl.storage_stop_event.set()


# --------------------------------------------------------------------------- #
# Scenario B: the "prefetch_thread" background thread -> mooncake batch_exists
# --------------------------------------------------------------------------- #
def scenario_b():
    print("\n\n" + "#" * 100)
    print("# SCENARIO B : thread that calls mooncake batch_exists -> batch_is_exist")
    print("#" * 100 + "\n")
    mem_pool_host = FakeHostKVCache()
    backend = make_mooncake_store(mem_pool_host)
    ctrl = make_controller(backend, mem_pool_host)

    op = PrefetchOperation(
        request_id="req-B",
        host_indices=mem_pool_host.alloc(N_PAGES * PAGE_SIZE),
        token_ids=list(range(N_PAGES * PAGE_SIZE)),
        last_hash=None,
        prefix_keys=None,
    )
    ctrl.prefetch_queue.put(op)

    t = threading.Thread(
        target=HiCacheController.prefetch_thread_func,
        args=(ctrl,),
        name="prefetch_thread",
        daemon=True,
    )
    t.start()
    backend.store.done.wait(timeout=10)
    time.sleep(0.3)
    ctrl.storage_stop_event.set()


# --------------------------------------------------------------------------- #
# Scenario C: scheduler main thread -> prefetch_from_storage -> submit
# --------------------------------------------------------------------------- #
def scenario_c():
    print("\n\n" + "#" * 100)
    print("# SCENARIO C : scheduler main thread submits the prefetch")
    print("#" * 100 + "\n")
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    mem_pool_host = FakeHostKVCache()
    backend = make_mooncake_store(mem_pool_host)
    ctrl = make_controller(backend, mem_pool_host)

    # instrumentation only: wrap the real method, then call the original
    orig_prefetch = HiCacheController.prefetch

    def traced_prefetch(self, request_id, host_indices, new_input_tokens,
                        last_hash=None, prefix_keys=None):
        dump("SCHEDULER MAIN THREAD  ->  HiCacheController.prefetch() (submit)")
        return orig_prefetch(self, request_id, host_indices, new_input_tokens,
                             last_hash, prefix_keys)

    HiCacheController.prefetch = traced_prefetch
    try:
        tree = object.__new__(HiRadixCache)
        tree.page_size = PAGE_SIZE
        tree.enable_storage = True
        tree.prefetch_threshold = 1
        tree.hicache_storage_pass_prefix_keys = False
        tree.ongoing_prefetch = {}
        tree.cache_controller = ctrl

        class FakeNode:
            def protect_host(self):
                pass

            def release_host(self):
                pass

        tree.evict_host = lambda n: None
        HiRadixCache.prefetch_from_storage(
            tree,
            req_id="req-C",
            last_host_node=FakeNode(),
            new_input_tokens=list(range(N_PAGES * PAGE_SIZE)),
            last_hash=None,
            prefix_keys=None,
        )
    finally:
        HiCacheController.prefetch = orig_prefetch


# --------------------------------------------------------------------------- #
# Scenario D: the deprecated generic path -> mooncake batch_get
# --------------------------------------------------------------------------- #
def scenario_d():
    print("\n\n" + "#" * 100)
    print("# SCENARIO D : deprecated generic path -> mooncake batch_get")
    print("#" * 100 + "\n")
    mem_pool_host = FakeHostKVCache()
    backend = make_mooncake_store(mem_pool_host)
    ctrl = make_controller(backend, mem_pool_host)
    ctrl.page_get_func = ctrl._generic_page_get

    op = PrefetchOperation(
        request_id="req-D",
        host_indices=mem_pool_host.alloc(N_PAGES * PAGE_SIZE),
        token_ids=list(range(N_PAGES * PAGE_SIZE)),
        last_hash=None,
        prefix_keys=None,
    )
    op.hash_value = [f"hash_{i}" for i in range(N_PAGES)]

    ctrl.prefetch_buffer.put(op)

    t = threading.Thread(
        target=HiCacheController.prefetch_io_aux_func,
        args=(ctrl,),
        name="prefetch_io_aux",
        daemon=True,
    )
    t.start()
    backend.store.done.wait(timeout=10)
    time.sleep(0.3)
    ctrl.storage_stop_event.set()


if __name__ == "__main__":
    faulthandler.enable()
    scenario_a()
    scenario_b()
    scenario_c()
    scenario_d()
    print("\nDONE")

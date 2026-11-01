# sglang HiCache + Mooncake Prefetch 性能分析

> 分析对象：线上 GLM-5.x（MLA）部署，sglang v0.5.9 + Mooncake Storage Backend
> 分析环境：WSL `/home/mzygd/workspace/sglang`（v0.5.9, detached HEAD）+ Mooncake C++ 源码 `/mnt/d/Codes/Mooncake`
> 日期：2026-11-01
> 触发问题：**「`_page_transfer` 很快，但整个 `PrefetchOperation` 的耗时很高」**

---

## 0. 结论摘要（TL;DR）

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | 调用 mooncake `batch_get_v1` 的是后台线程 **`prefetch_io_aux_thread`**，不是 scheduler 主线程 | 实跑抓栈 |
| 2 | **MLA 下 `batch_is_exist` 与 `batch_get_into` 的 RPC 次数严格 1:1**，单次 exists 更便宜 → IO 本身不是瓶颈 | 实测 + 源码 |
| 3 | **真正的瓶颈是排队**：`prefetch_thread` / `prefetch_io_aux_thread` 各只有 1 个线程、两级串行 → head-of-line blocking。实测 256 并发时排队等待涨到 **1.45 秒**，而 `_page_transfer` 自身恒定 **3.6~5.6 ms** | 实测 |
| 4 | **收割（harvest）被 `batch_is_full` 卡住**：唯一收割点 `scheduler.py:2086` 位于 staging loop 内，而该 loop 在 batch 满时会 `break`（:2081）或整个 early-return（:1993）→ **底层早完成了，radix tree 索引却迟迟建不上** | 源码 + 实测 |
| 5 | **收割周期 ≈ 一次 forward 的墙钟 T**：overlap 调度器里 scheduler 线程阻塞在 `pop_and_process()` → `copy_done.synchronize()`（等上一批结果拷回 CPU），不是空轮询 | 源码 |
| 6 | **在收割到来之前，数据物理上已在 host 内存里（零拷贝直写），但 radix tree 里没有索引 → 请求 `match_prefix` 命中不了 → L3 收益为 0，且 host 页与 `prefetch_tokens_occupied` 一直被占着** | 源码 |
| 7 | `prefetch_tokens_occupied` 只在收割/abort 时递减；`SGLANG_REQ_WAITING_TIMEOUT` 默认 -1 不启用 → **收割不来 → 占用不释放 → `prefetch_rate_limited()` 触发 → 新请求的 prefetch 被直接拒 → 整套机制停摆** | 源码 |
| 8 | mooncake 侧：`batch_is_exist` 和 `batch_get_into` **都打 master**；差别是 Exist **每个 key 都要写 metadata 续租**，GetReplicaList 是纯读 | Mooncake 源码 |
| 9 | MLA（page_size=64）1 次 RPC / 8192 token；**MHA（page_size=1，默认）会把 RPC 次数放大 64 倍** | 源码 + 实测 |
| 10 | `operation.host_indices is None` 分支在 v0.5.9 **是死代码**（三处），实际"host 内存不够"表现为「请求压根没进 `ongoing_prefetch`」 | 源码 |

---

## 1. 环境与版本

| 项 | 值 |
|---|---|
| sglang 仓库 | `/home/mzygd/workspace/sglang`（v0.5.9, detached HEAD @ `bbe9c7ee`） |
| Python | 3.12 |
| torch | 2.9.1+cu128 |
| Mooncake C++ 源码 | `/mnt/d/Codes/Mooncake`（Windows 侧 `D:\Codes\Mooncake`） |
| 本机 GPU | RTX 3070 (sm_86)；线上 MLA 走 FlashMLA(page_size=64) 或 Cutlass MLA(page_size=128) |
| WSL 挂载 | `D:` → `/mnt/d`，`C:` → `/mnt/c` |

> ⚠️ 本机 WSL **未安装 mooncake**（`import mooncake` → ModuleNotFoundError）。所有 IO 行为通过替换 `MooncakeDistributedStore` 为可控延迟的替身来实测；**RPC 次数与 Python 开销是代码决定的精确值，延迟配比是建模值**。

---

## 2. 线程布局：谁在调 mooncake

### 2.1 三个后台线程 + scheduler 主线程

| 线程 | 创建位置 | 职责 | mooncake API |
|---|---|---|---|
| `prefetch_thread` | `cache_controller.py:337` (`_start_storage_threads`) | 探测 storage 命中 | `batch_is_exist` |
| **`prefetch_io_aux_thread`** | `cache_controller.py:915`（在 `prefetch_thread_func` 内部再起） | **真正把 KV 拉进 host 内存** | **`batch_get_into`** |
| `backup_thread` | `cache_controller.py:340` | 写回 storage | `batch_put_from` |
| scheduler 主线程 | — | 提交 prefetch / 收割结果 | 不直接调 mooncake |

### 2.2 真实 backtrace（faulthandler 格式，实跑抓取）

**A. `batch_get_v1` → `batch_get_into`（在 `prefetch_io_aux_thread` 上）**

```
Current thread 0x0000724f0fdf86c0 (most recent call first):
  File ".../mooncake_store/mooncake_store.py", line 722 in _get_batch_zero_copy_impl
    return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
  File ".../mooncake_store/mooncake_store.py", line 518 in batch_get_v1
    get_results = self._get_batch_zero_copy_impl(key_strs, buffer_ptrs, buffer_sizes)
  File ".../managers/cache_controller.py", line 792 in _page_get_zero_copy
    results = self.storage_backend.batch_get_v1(hash_values, host_indices, extra_info)
  File ".../managers/cache_controller.py", line 839 in _page_transfer
    self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
  File ".../managers/cache_controller.py", line 860 in prefetch_io_aux_func
    self._page_transfer(operation)
  File "/usr/lib/python3.12/threading.py", line 1010 in run
  File "/usr/lib/python3.12/threading.py", line 1073 in _bootstrap_inner
  File "/usr/lib/python3.12/threading.py", line 1030 in _bootstrap
```

**B. `batch_exists` → `batch_is_exist`（在 `prefetch_thread` 上）**

```
Current thread 0x0000724f0f5f76c0 (most recent call first):
  File ".../mooncake_store/mooncake_store.py", line 725 in _batch_exist
    return self.store.batch_is_exist(key_strs)
  File ".../mooncake_store/mooncake_store.py", line 700 in batch_exists
    exist_result = self._batch_exist(query_keys)
  File ".../managers/cache_controller.py", line 900 in _storage_hit_query
    hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
  File ".../managers/cache_controller.py", line 924 in prefetch_thread_func
    hash_value, storage_hit_count = self._storage_hit_query(operation)
  File "/usr/lib/python3.12/threading.py", line 1010 in run
  ...
```

**C. scheduler 主线程提交**

```
Current thread 0x00007250ba2fa140 (most recent call first):
  File ".../managers/cache_controller.py", line 772 in prefetch
    self.prefetch_queue.put(operation)
  File ".../mem_cache/hiradix_cache.py", line 1192 in prefetch_from_storage
    operation = self.cache_controller.prefetch(req_id, host_indices, new_input_tokens, last_hash, prefix_keys)
```

### 2.3 Prefetch 流水线全貌

```
scheduler 主线程
   │  prefetch_queue.put(op)        ← 所有请求塞进同一个 FIFO
   ▼
prefetch_thread        ★ 只有 1 个线程
   │  ① _storage_hit_query → batch_exists → master 读 + 写续租
   │  ② prefetch_buffer.put(op)
   ▼
prefetch_io_aux_thread ★ 只有 1 个线程
   │  ③ _page_transfer → batch_get_into → master 读 + segment RDMA（零拷贝直写 host buffer）
   │  ④ ★整个 operation 的所有批次拉完，才取下一个 op
   ▼
done
```

**两级都单线程串行 ⇒ 流水线吞吐上限 ≈ 1 / 单个 op 的串行时间。**

---

## 3. 核心问题：为什么「page_transfer 很快，整体很慢」

### 3.1 实测：排队等待 vs `_page_transfer` 自身耗时

条件：MLA `page_size=64`，8192 tokens/req，mooncake `exists=2ms/call`、`get=2ms/call`（两者对等，无延迟假设）

| 并发 reqs | 全部完成耗时 | **排队等待 首→末** | **`_page_transfer` 自身 均值** |
|---|---|---|---|
| 1 | 9.0 ms | 4.6 ms | **3.64 ms** |
| 4 | 26.5 ms | 6.1 → 21.3 ms | **4.69 ms** |
| 16 | 83.6 ms | 7.2 → 78.0 ms | **4.63 ms** |
| 64 | 328.8 ms | 7.0 → 324.3 ms | **4.96 ms** |
| 256 | 1453.0 ms | 5.6 → **1448.3 ms** | **5.59 ms** |

**⇒ `_page_transfer` 恒定 3.6~5.6 ms（"很快"），但排队等待随负载线性涨到 1.45 s（"很慢"）。**

**⇒ 盯着 `_page_transfer` 看永远看不到瓶颈；要看 `prefetch_queue.qsize()` / `prefetch_buffer.qsize()`。**

### 3.2 零网络延迟下的纯 Python/结构开销

| tokens | `_storage_hit_query` | `_page_transfer` | exists RPC 次数 | get RPC 次数 |
|---|---|---|---|---|
| 4096 | 6.26 ms | 5.82 ms | 32 | 32 |
| 16384 | 22.04 ms | 15.45 ms | 128 | 128 |
| 65536 | 84.64 ms | 67.00 ms | 512 | 512 |

**⇒ RPC 次数 = tokens / 128（page_size=1 时）。65K tokens → 512 次串行 RPC。**
**⇒ page_size=1 时每个 token 一次链式 SHA-256，65K tokens ≈ 85 ms 纯 Python，跑在唯一的 `prefetch_thread` 里。**

### 3.3 MLA vs MHA：page_size 决定了 RPC 次数

| page_size | exists 次数 | get 次数 | 端到端（4096 tokens） |
|---|---|---|---|
| **1**（MHA 默认，`server_args.py:2055`） | 32 | 32 | **102 ms** |
| **64**（MLA / FlashMLA） | 1 | 1 | **3.9 ms** |

**⇒ 26 倍差距。通用 MHA 模型 page_size=1 会把 RPC 次数放大 64 倍。**

---

## 4. Scheduler 主循环与收割点

### 4.1 overlap 模式主循环（默认，`disable_overlap_schedule=False`，`server_args.py:596`）

```python
def event_loop_overlap(self):                        # scheduler.py:1138
    while True:
        recv_reqs = self.recv_requests()             # ① zmq.NOBLOCK 排空 → 不阻塞
        self.process_input_requests(recv_reqs)        # ② 提交 prefetch → 不阻塞
        batch = self.get_next_batch_to_run()          # ③ ★收割在这里★ 纯 CPU，不等 GPU
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)
        if disable_overlap_for_batch:
            pop_and_process()                         # ③'
        if batch:
            batch_result = self.run_batch(batch)      # ④ launch forward → 异步，立即返回
            self.result_queue.append((batch.copy(), batch_result))
        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()                     # ⑤ ★★★真阻塞在这里★★★
        elif batch is None:
            self.self_check_during_idle()
        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result)
        self.last_batch = batch
```

**⑤ 的阻塞链路：**

```
pop_and_process()
 → process_batch_result()                          scheduler.py:2447
   → process_batch_result_decode()                 scheduler_output_processor_mixin.py:417
     → result.copy_done.synchronize()              scheduler_output_processor_mixin.py:422-423  ★阻塞★
```

`copy_done` 的 record 点：`scheduler.py:2333`（`Event()`）+ `managers/utils.py:78`（`copy_done.record()`，在 `copy_to_cpu(non_blocking=True)` 之后）。

**⇒ scheduler 线程不是空轮询，它阻塞在 ⑤ 等上一批的 output token 拷回 CPU。**

### 4.2 收割点被两道 batch 门卡住

**门 1 —— early return（`scheduler.py:1990-1993`）**

```python
if self.try_preemption:
    self.running_batch.batch_is_full = False        # :1988 只有开了 priority scheduling 才每轮重置

if (
    self.running_batch.batch_is_full or len(self.waiting_queue) == 0
) and self.chunked_req is None:
    return None                                     # :1993 ★staging loop 都不进★
```

**门 2 —— staging loop 里的 break（`scheduler.py:2079-2083`）**

```python
for req in self.waiting_queue:
    ...
    if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
        self.running_batch.batch_is_full = True     # :2072
    if self.running_batch.batch_is_full:
        if not self.try_preemption or not adder.preempt_to_schedule(req, self.server_args):
            break                                   # :2081 ★跳出整个 loop★
    if self.enable_hicache_storage:
        prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)   # :2086 ★唯一收割点★
```

### 4.3 `batch_is_full` 什么时候被清掉

**只有一个地方**（`scheduler.py:2262-2263`）：

```python
def update_running_batch(self, batch):
    ...
    if batch.batch_size() < initial_bs:
        batch.batch_is_full = False     # ★只有当这一轮 decode 有请求结束、batch 变小了★
```

而 `update_running_batch` **只在 `get_new_batch_prefill()` 返回 None 时才被调用**（`scheduler.py:1938-1941`）：

```python
new_batch = self.get_new_batch_prefill()
if new_batch is not None:
    ret = new_batch
else:
    if not self.running_batch.is_empty():
        self.running_batch = self.update_running_batch(self.running_batch)
```

**⇒ 自我强化的闭环：**

```
batch 填满 → batch_is_full = True
   └─► get_new_batch_prefill() early return → staging loop 不进 → 不收割
          └─► 走 decode 分支 → update_running_batch()
                 └─► 只有本轮有请求结束，才清 batch_is_full
                        └─► 下一轮 staging loop 才进得去 → 才收割
```

**⇒ 「有空位」= 「上一轮 decode 有请求结束」；收割频率 = 请求结束频率，而不是 scheduler 迭代频率。**

### 4.4 收割周期 ≈ 一次 forward 的墙钟 T

| | overlap（默认） | 非 overlap（`event_loop_normal` :1108） |
|---|---|---|
| ④ `run_batch` | 异步 launch，立即返回 | **同步阻塞**到 forward 完 |
| ⑤ | 阻塞等**上一批**的 copy_done | 无 ⑤（④ 自己就阻塞） |
| ③ 收割时机 | 在**上一个** forward 还在跑时 | 在上一个 forward **结束后** |
| **收割周期** | **≈ T** | **≈ T** |

**⇒ overlap 只是把 ③ 挪进 GPU 工作窗口（提吞吐），并没有缩短收割周期。**

### 4.5 为什么各 TP rank 的 break 点会不一样（TP>1 隐患）

`break` 的触发条件依赖 `can_run_list`，而 `can_run_list` 装不装得下取决于**本地状态**：

```python
# schedule_policy.py:448-467  (@property rem_total_tokens)
available_and_evictable = (
    self.token_to_kv_pool_allocator.available_size()
    + self.tree_cache.evictable_size()      # ★本地 radix tree 状态★
)
```

- `tree_cache.evictable_size()` —— 各 rank 的树状态/驱逐时机/命中历史不同 → 数值不同
- `token_to_kv_pool_allocator.available_size()` —— MHA 下 KV pool 按 TP 分片，各 rank 容量不同（MLA 下是复制的，这条不成立）

---

## 5. 收割前 vs 收割后：三重损失

**零拷贝意味着 mooncake 把 KV 直接写进 host KV pool 的注册 buffer**
（`mooncake_store.py:718 _get_batch_zero_copy_impl` ← `mem_pool_host.get_page_buffer_meta()` 给的裸指针；`register_buffer` 见 `mooncake_store.py:447`）

**⇒ 在收割到来之前：**

| 状态 | 结果 |
|---|---|
| ✅ 数据物理上在 host 内存里 | 已经付了 mooncake 的 IO 成本 |
| ❌ radix tree 里没有索引（`_insert_helper_host` 没执行） | 后续请求 `match_prefix` 命中不了 → **L3 收益 = 0** |
| ❌ `req.storage_hit_length` 没设 | 请求的 token 账算不对 |
| ❌ host 页被 `alloc` 占着不释放，`prefetch_tokens_occupied` 也不减 | 内存泄漏式占用 + 限流反馈环 |

**⇒ 内存占着 + 收益归零 —— 直到收割那一刻才翻转。**

### 5.1 限流反馈环（死循环）

`prefetch_tokens_occupied` **只在三处递减**：

| 位置 | 场景 |
|---|---|
| `hiradix_cache.py:1098` | `check_prefetch_progress`（收割） |
| `hiradix_cache.py:412 / 473` | drain revoke / backup 队列 |
| `hiradix_cache.py:1394` | `release_aborted_request`（请求被 abort） |

`SGLANG_REQ_WAITING_TIMEOUT` 默认 **-1（不启用）**（`environ.py:259`）→ 等太久的请求永远不会被 abort → 占用永不释放。

```
收割被 batch_full 的 break 跳过
  → prefetch_tokens_occupied 居高不下
    → cache_controller.py:873  prefetch_rate_limited() 返回 True
      → prefetch_from_storage 提前 return，新请求的 prefetch 被直接拒
        → 整套 prefetch 机制停摆
```

### 5.2 `best_effort` 策略：收割太早 = 数据被丢弃

`hicache_storage_prefetch_policy` 默认 `best_effort`（`server_args.py:536`）→ `can_terminate_prefetch`（`hiradix_cache.py:1007`）恒返回 True → **第一次 check 就 `mark_terminate()` 冻结 `completed_tokens`**。

实测（MLA, 4096 tokens, exists/get 各 2ms）：

| 首次 check 延迟 | exists | get | **交付 token** |
|---|---|---|---|
| 0.0 ms | 1 | 1 | **0** |
| 2.0 ms | 1 | 1 | **0** |
| 5.0 ms | 1 | 1 | **0** |
| 20.0 ms | 1 | 1 | **4096** ✓ |
| 100.0 ms | 1 | 1 | **4096** ✓ |

**⇒ 收割早于 IO 完成 → `increment()` 因已 terminate 返回 False → `completed_tokens` 不记账 → `_page_transfer` 检测到不匹配就 break → `append_host_mem_release(operation.host_indices[completed_tokens:])` 把刚拉回来的整批 host 页全部释放 → IO 成本照付、收益归零。**

---

## 6. Mooncake C++ 侧证据

### 6.1 两条路径都打 master

| Python API | C++ 链路 |
|---|---|
| `batch_is_exist` | `client_service.cpp:3228 Client::BatchIsExist` → `master_client_.BatchExistKey(keys)` → master `master_service.cpp:2318 BatchExistKey` |
| `batch_get_into` | `real_client.cpp:4396 RealClient::batch_get_into` → `:4671 batch_get_into_internal` → `client_->BatchQuery(keys)` → `client_service.cpp:1116 master_client_.BatchGetReplicaList(...)` → master `master_service.cpp:3113` |

### 6.2 关键差异：Exist 是「读 + 写」，GetReplicaList 是「纯读」

```cpp
// master_service.cpp:2318  BatchExistKey —— 每个 key：
    GrantLeaseForGroup(tenant_state, key, metadata);      // ★写 metadata：续租 + soft pin★
```

```cpp
// master_service.cpp:2012  GrantLeaseForGroup
    if (!metadata.IsGrouped()) { metadata.GrantLease(ttl, soft_pin_ttl); return; }   // 写
    ...
    for (const auto& member_key : group_it->second) {                              // ★grouped key：
        mit->second.GrantLease(...);                                               // 遍历整个 group 逐个续租
    }
```

而 `BatchGetReplicaList`（`master_service.cpp:3113`）是同样的 shard 分组 + `snapshot_mutex_` shared_lock，但**纯读、不续租**。

**⇒ 「exists 只是查在不在所以应该更快」在抽象层对，在 mooncake 实现层不成立。**

### 6.3 两者的 shard 遍历成本相同

```cpp
// BatchExistKey :2318 和 BatchGetReplicaList :3113 都是：
const size_t start_shard = randomIndex(kNumShards);
for (size_t scanned = 0; scanned < kNumShards; ++scanned) {
    const size_t shard_idx = (start_shard + kNumShards - scanned) % kNumShards;
    ...
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    MetadataShardAccessorRO shard(this, shard_idx);
    ...
}
```

**⇒ key 分散在多个 shard 上时，两者都要多次取锁 + 构造 accessor，差异只在续租写。**

---

## 7. MLA vs MHA 的完整差异

| 维度 | MLA（GLM/DeepSeek 系） | MHA（通用模型） |
|---|---|---|
| `page_size` | 64（FlashMLA）/ 128（Cutlass MLA） | **1**（`server_args.py:2055` 默认） |
| `is_mla_backend` | True | False |
| key 数 / 页 | **1**（不拆 k/v） | 2（`_k` / `_v`，`mooncake_store.py:464-469`） |
| `_get_*_buffer_meta` | 1 ptr/页 | 2 ptr/页 |
| exists 分片 | `page_size × storage_batch_size` = **8192 token** | `1 × 128` = **128 token** |
| 4096 token 的 RPC 数 | **1 exists + 1 get** | **32 exists + 32 get** |
| 131072 token 的 RPC 数 | 16 + 16 | 1024 + 1024 |
| KV pool 是否按 TP 分片 | 否（复制） | 是 |

`storage_batch_size = 128`（**页数**，硬编码于 `cache_controller.py:455`）

**⇒ 通用 MHA 模型 page_size=1 会把 RPC 次数放大 64 倍。**

---

## 8. 已知问题清单

| # | 问题 | 位置 | 状态 |
|---|---|---|---|
| 1 | `operation.host_indices is None` 分支**不可达**（死代码）。"host 内存不够"的真实表现是「请求压根没进 `ongoing_prefetch`」 | `hiradix_cache.py:1057` / `:1114` / `:1385` | 死代码 ×3 |
| 2 | `MooncakeStore.batch_get`（非 v1）在 prefetch 路径上必崩：`_generic_page_get` 只传 2 个参数，`batch_get` 断言三个长度 → `TypeError: object of type 'NoneType' has no len()` | `cache_controller.py:810` → `mooncake_store.py:653` | mooncake 恒走 v1（`:479` 强制 `_page_get_zero_copy`），此路不可达 |
| 3 | `prefix_keys` 在 mooncake 后端**完全没被读取**（只在 `HiCacheStorageExtraInfo` 定义），但 `_storage_hit_query`/`_page_transfer` 里 `prefix_keys += batch_hashes` 是 **O(n²)** 拷贝 | `cache_controller.py:899/848` | 默认 `hicache_storage_pass_prefix_keys=False` 才无害 |
| 4 | `best_effort` 下第一次 check 就冻结 `completed_tokens` → 收割早 = 数据被丢弃 | `hiradix_cache.py:1007` | 默认策略 |
| 5 | `prefetch_from_storage` 里 `alloc` 失败会在 **scheduler 主线程上同步 `evict_host(prefetch_length)`** | `hiradix_cache.py:1185-1187` | host 内存紧张时的隐藏慢路径（发生在 `start_time` 之前，不计入 op 耗时，但会拖慢 scheduler） |
| 6 | **TP>1 集合通信错位 → 挂死** | `hiradix_cache.py:1034/1075/986/688` | 见下 |

### 8.1 TP>1 的 hang 机制

`hiradix_cache.py` 里 4 个 `all_reduce`（都在 `self.tp_group` = gloo cpu_group 上）：

| 位置 | 内容 |
|---|---|
| `:1034` | `all_reduce(MAX)` — states `[1-can_terminate, terminated]` |
| `:1075` | `all_reduce(MIN)` — `completed_tokens` |
| `:986` | `all_reduce(MIN)` — 三个队列的 qsize |
| `:688` | `all_reduce(MIN)` |

**而 `check_prefetch_progress`（`:1046`）在这两个分支提前 return，★不做任何 all_reduce★：**

```python
if req_id not in self.ongoing_prefetch:     # :1047-1049
    return True
if operation.host_indices is None:           # :1057
    return True
```

**⇒ 同一个 req_id 在不同 rank 上的 all_reduce 次数可以是 0 / 1 / 2 三种。**

gloo 按**调用顺序**配对、不校验内容：

| rank 0 序列 | rank 1 序列 | 结果 |
|---|---|---|
| #1 can_terminate(A) | #1 can_terminate(B) | 形状/dtype 相同 → **不报错，静默算成 A∪B 的 MAX** |
| #2 min_completed(A) | #2 min_completed(B) | **静默算出 min(A, B)** |
| #3 can_terminate(B) | — | **无配对 → 阻塞** |
| #4 min_completed(B) | — | 阻塞 |

**⇒ 先静默算错（`min_completed_tokens` 偏小 → 少插 token / KV 不一致），再硬挂死。**
**⇒ 30 分钟后 gloo 抛 `RuntimeError`（`torch.distributed.constants.default_pg_timeout = 0:30:00`）→ scheduler 进程崩溃，无法恢复，只能重启。**

**线上不卡死 ⇒ 大概率是 TP=1**（`tp_world_size == 1` 时这 4 个 all_reduce 全不执行）。

---

## 9. 修复方案（按优先级）

### P0 — 打破收割 gate（治「已完成但未收割」）

在 `scheduler.py:1993` 的 early return **之前**加一次独立收割：

```python
# scheduler.py, _get_new_batch_prefill_raw 开头
if self.enable_hierarchical_cache:
    self.tree_cache.check_hicache_events()
    self.tree_cache.harvest_all_prefetch()      # ← 新增，与 batch_is_full 完全解耦
```

```python
# hiradix_cache.py 新增
def harvest_all_prefetch(self):
    """收割所有 ongoing prefetch。不依赖 batch 排布。"""
    for rid in sorted(self.ongoing_prefetch.keys()):
        try:
            self.check_prefetch_progress(rid)
        except Exception:
            logger.exception("harvest_all_prefetch failed rid=%s", rid)
```

**同时必须把 `scheduler.py:2086` 那次逐请求调用去掉**（否则重复收割；TP>1 时第二次调用会跳过 all_reduce → 集合通信错位 → 挂死）。

> **TP=1**：直接挪，安全。
> **TP>1**：必须把 `check_prefetch_progress` 拆成「本地计算版」+「批量同步版」，**每轮固定 2 次 all_reduce**（与请求数量彻底解耦），保证各 rank 调用序列长度恒定。

### P1 — 提高流水线并发（治排队）

`cache_controller.py:329 _start_storage_threads` 现在是 1+1 个线程。优先加 `prefetch_io_aux_thread`（它一个 op 要拉完所有批次才放行下一个，最容易堵）：

```python
self.prefetch_io_aux_threads = [
    threading.Thread(target=self.prefetch_io_aux_func, daemon=True)
    for _ in range(self.prefetch_io_aux_worker_count)   # 建议 2~4
]
```

- `prefetch_io_aux_func` **没有集合通信** → TP=1 / TP>1 都能安全加（`Queue.get()` 保证每个 op 只被一个线程取走）
- ⚠️ `prefetch_thread` 里有 `all_reduce(MIN)`（`cache_controller.py:924`），**TP>1 时不能随便加它的并发**
- `_stop_storage_threads`（`:353`）里要同步 join 新增线程

### P2 — 增大 `storage_batch_size`（减少每个 op 的 RPC 次数）

`cache_controller.py:455` 硬编码 `128`（页数）→ 512 / 1024。

- MLA：8192 token 一次 RPC → 32768 token 一次 RPC
- ⚠️ 一次 RPC 的 key 数变多，master 端单次处理时间变长；需要实测找拐点

### P3 — 合并 exists + get（收益有上限，且有语义风险）

mooncake 的 `batch_get_into` 对不存在的 key 返回**负数**（`mooncake_store.py:485` 注释：*each element is the number of bytes read on success, or a negative value on error*），所以可以一次拉完再判断命中。

**但实测显示单纯省掉 exists RPC 收益有限**：把 exists 延迟设成 0，吞吐几乎没变（226 → 187 req/s），因为 `_storage_hit_query` 里的 Python 哈希 + 列表构建成本 ≈ RPC 成本。真要省必须**连哈希一起省**（挪到传输阶段复用，或缓存 page hash）。

**⚠️ 语义风险**：exists 会**续租**（写 metadata），get 不会。合并后**续租语义丢失**，对象可能在传输过程中被回收。**必须与 mooncake 上游确认后再改。**

### P4 — 调整 prefetch 策略

```bash
--hicache-storage-prefetch-policy timeout     # 默认 best_effort
```

`prefetch_timeout_base` 默认 1s，`prefetch_timeout_per_ki_token` 默认 0.25s（`hiradix_cache.py:543-547`）。

**只能治「收割太早」，治不了「收割不来」。真正对症的是 P0。**

### P5 — 兜底：设 `SGLANG_REQ_WAITING_TIMEOUT`

```bash
export SGLANG_REQ_WAITING_TIMEOUT=30   # 秒
```

让等太久的请求被 abort → `release_aborted_request` 释放占用 → **避免限流反馈环永久化**。

---

## 10. 线上打点清单（10 分钟定位）

| # | 打点位置 | 打什么 | 判据 |
|---|---|---|---|
| 1 | `cache_controller.py:857` (`prefetch_io_aux_func`) | `self.prefetch_buffer.qsize()` | **持续增长不回落 → 流水线饱和（P1）** |
| 2 | `cache_controller.py:921` (`prefetch_thread_func`) | `self.prefetch_queue.qsize()` | 同上 |
| 3 | `cache_controller.py:873` (`prefetch_rate_limited`) | 返回 True 时 warn | 频繁触发 → 已停摆 |
| 4 | `hiradix_cache.py:1046` (`check_prefetch_progress` 入口) | `age` = `now - operation.start_time`；`completed_tokens` / 总数 | **age 大而 completed 已满 → 收割被 gate（P0）** |
| 5 | `hiradix_cache.py:1185` (`prefetch_from_storage`) | `evict_host()` 耗时 | 耗时可观 → host 内存不足，scheduler 被同步驱逐拖慢 |

**建议的 harvest 日志：**

```python
logger.info("[hicache] harvest req=%s age=%.1fms done=%d/%d occupied=%d",
            req_id,
            (time.monotonic() - operation.start_time) * 1000,
            operation.completed_tokens,
            len(operation.hash_value) * self.page_size,
            self.cache_controller.prefetch_tokens_occupied)
```

**⇒ 三个候选 slow 源，用上表区分开：**

| # | 候选 slow 源 | 特征 | 对应修复 |
|---|---|---|---|
| 1 | **流水线排队** | qsize 持续增长 | P1（加 worker）+ P2 |
| 2 | **收割被 gate** | age 大、`completed_tokens` 已满 | P0（打破 gate） |
| 3 | **host 内存不足导致同步驱逐** | `evict_host` 耗时可观 | 调 `--hicache-ratio` / 扩容 host 内存 |

---

## 11. 附：探针脚本

下列脚本是本地分析用的探针工具，**未包含在本仓库中**（路径为本机 Windows 环境）。跑法：

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/mzygd/workspace/sglang && .venv/bin/python /mnt/d/Codes/<脚本名>"
```

| 脚本 | 作用 |
|---|---|
| `mooncake_backtrace_repro.py` | 抓取调用 mooncake `batch_get_xx` 的线程真实 backtrace（faulthandler 格式） |
| `prefetch_latency_probe.py` | MHA/page_size=1 的分阶段耗时拆解 |
| `prefetch_mla_probe.py` | MLA 版：RPC 次数表 + 收割时机实验 + 并发场景 |
| `prefetch_interrupt_probe.py` | 专门验证「收割时机决定收益」 |
| `prefetch_queue_probe.py` | **流水线排队实测**（并发压力下 `_page_transfer` vs 排队等待） |

> 所有脚本都用真实 sglang 代码路径，只把 mooncake 的 C++ 对象 `MooncakeDistributedStore` 换成可控延迟的替身。**RPC 次数、Python 开销是代码决定的精确值；延迟配比是建模值。**

---

## 12. 关键文件行号索引

### sglang
| 内容 | 位置 |
|---|---|
| `_start_storage_threads`（起 3 个后台线程） | `python/sglang/srt/managers/cache_controller.py:329` |
| `_stop_storage_threads` | `cache_controller.py:353` |
| `storage_batch_size = 128`（硬编码） | `cache_controller.py:455` |
| `prefetch_capacity_limit` | `cache_controller.py:452` |
| `prefetch()`（只是 put 队列） | `cache_controller.py:761` |
| `_page_get_zero_copy` | `cache_controller.py:789` |
| `_generic_page_get`（非 v1，会崩） | `cache_controller.py:806` |
| `_page_transfer` | `cache_controller.py:828` |
| `prefetch_io_aux_func` | `cache_controller.py:851` |
| `prefetch_rate_limited` | `cache_controller.py:868` |
| `_storage_hit_query` | `cache_controller.py:878` |
| `prefetch_thread_func` | `cache_controller.py:910` |
| `PrefetchOperation`（`start_time` / `increment` / `mark_terminate`） | `cache_controller.py:213-244` |
| `MooncakeStore.batch_get_v1` | `mem_cache/storage/mooncake_store/mooncake_store.py:506` |
| `MooncakeStore.batch_get`（会崩） | `mooncake_store.py:647` |
| `MooncakeStore.batch_exists` | `mooncake_store.py:682` |
| `_get_batch_zero_copy_impl` | `mooncake_store.py:719` |
| `_batch_postprocess`（负数=不存在） | `mooncake_store.py:485` |
| `register_buffer` | `mooncake_store.py:447` |
| `HiRadixCache.check_prefetch_progress` | `mem_cache/hiradix_cache.py:1046` |
| `can_terminate_prefetch`（best_effort 恒 True） | `hiradix_cache.py:1007` |
| `release_aborted_request` | `hiradix_cache.py:1377` |
| `prefetch_from_storage` | `hiradix_cache.py:1163` |
| `drain_storage_control_queues` | `hiradix_cache.py:970` |
| `scheduler._prefetch_kvcache` | `managers/scheduler.py:1656` |
| `scheduler.event_loop_overlap` | `managers/scheduler.py:1138` |
| `scheduler.event_loop_normal` | `managers/scheduler.py:1108` |
| `scheduler.get_next_batch_to_run` | `managers/scheduler.py:1875` |
| 收割点（唯一） | `managers/scheduler.py:2086` |
| batch 门 1（early return） | `managers/scheduler.py:1990-1993` |
| batch 门 2（break） | `managers/scheduler.py:2079-2083` |
| `batch_is_full` 唯一清除点 | `managers/scheduler.py:2262-2263` |
| `copy_done.synchronize()`（真阻塞） | `managers/scheduler_output_processor_mixin.py:422-423` |
| `hicache_storage_prefetch_policy` 默认 `best_effort` | `server_args.py:536` |
| `page_size` 默认 1 | `server_args.py:2055` |
| `SGLANG_REQ_WAITING_TIMEOUT` 默认 -1 | `environ.py:259` |

### Mooncake C++
| 内容 | 位置 |
|---|---|
| `Client::BatchIsExist` | `mooncake-store/src/client_service.cpp:3228` |
| `Client::BatchQuery` | `mooncake-store/src/client_service.cpp:1107` |
| `RealClient::batch_get_into_internal` | `mooncake-store/src/real_client.cpp:4644` |
| master `BatchExistKey` | `mooncake-store/src/master_service.cpp:2318` |
| master `BatchGetReplicaList` | `mooncake-store/src/master_service.cpp:3113` |
| `GrantLeaseForGroup`（写续租） | `mooncake-store/src/master_service.cpp:2012` |

---

## 13. 未验证 / 待确认

1. **线上 TP 是几？** 不卡死强烈暗示 TP=1（`tp_world_size == 1` 时 4 个 all_reduce 全不执行）。TP>1 则 P0 必须用「每轮固定 2 次 all_reduce」的批量同步版。
2. **线上是否启用了 overlap 调度？** `disable_overlap_schedule` 默认 False，但 mamba no_buffer / sparse-head embedding 会自动关。两种模式下收割周期都是 ≈ T，但非 overlap 时 CPU 完全不重叠。
3. **真实 mooncake 的 `batch_is_exist` vs `batch_get_into` 单次延迟比** —— 本文档用 2ms 对等建模。线上需要实测（第 6 节给出了「exists 带写续租、get 纯读」的机制解释，但实际比值取决于 master 负载与续租是否命中缓存）。
4. **真实 host 内存配置**（`--hicache-ratio` 之类）—— 决定第 8 节问题 5（同步驱逐）是否是主要慢源。
5. `prefix_keys` 合并方案（P3）的续租语义问题，需与 mooncake 上游确认。

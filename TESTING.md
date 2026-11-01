# 测试方法论：这批数据到底是怎么来的

> 配套文档：[sglang-hicache-mooncake-prefetch.md](./sglang-hicache-mooncake-prefetch.md)
> 日期：2026-11-01

---

## 0. 一句话

**本机没有 mooncake、没有模型、也没有跑真实推理。测试测的是 sglang 自己的控制流 —— mooncake 的 C++ 扩展和 GPU 内存池被替换成可控替身，sglang 的代码一行没改。**

---

## 1. 本机缺什么，怎么绕过去的

| 缺什么 | 实际影响 | 绕法 |
|---|---|---|
| **mooncake C++ 扩展**（WSL venv 里 `import mooncake` → ModuleNotFoundError） | 没法真的调 `batch_get_into` / `batch_is_exist` | 用替身实现**同样的调用契约**：入参形状、返回语义、延迟可注入 |
| **模型 / GPU 推理** | 没法用真实请求产生 prefetch | 不用模型：直接构造 `PrefetchOperation`，驱动真实的 `HiCacheController` / `HiRadixCache` 方法 |
| **多卡 / 分布式** | 没法验证 TP>1 的集合通信 | 固定 `tp_world_size=1` → `hiradix_cache.py` 里 4 个 `all_reduce` 分支天然不执行 |
| **真实 device KV pool** | `HiRadixCache.__init__` 起不来（要 device pool + server_args + 分布式初始化） | `object.__new__(HiRadixCache)` 跳过 `__init__`，再用 `__get__` 把**真实方法体**绑到桩实例上 |
| **真实 radix tree** | 插入会因缺字段报错 | 只桩掉 `_insert_helper_host`（返回 0）；插入正确性不在本次结论范围内 |

---

## 2. 什么是真的，什么是替身

### 2.1 真实代码（直接 import 自已安装的 sglang，一行没改）

| 模块 | 用到的真实成员 |
|---|---|
| `sglang.srt.managers.cache_controller` | `HiCacheController` 的**全部方法**：`_start_storage_threads` / `prefetch_thread_func` / `prefetch_io_aux_func` / `_storage_hit_query` / `_page_transfer` / `_page_get_zero_copy` / `prefetch` / `terminate_prefetch` / `prefetch_rate_limited`；`PrefetchOperation` 的**全部**：`start_time` / `increment` / `mark_terminate` / `is_terminated` |
| `sglang.srt.mem_cache.storage.mooncake_store.mooncake_store` | `MooncakeStore` 的**全部**：`batch_get_v1` / `batch_exists` / `_batch_preprocess` / `_batch_postprocess` / `_get_batch_zero_copy_impl` / `_batch_exist` |
| `sglang.srt.mem_cache.hiradix_cache` | `HiRadixCache.prefetch_from_storage` / `check_prefetch_progress` / `can_terminate_prefetch` |
| `sglang.srt.mem_cache.hicache_storage` | `get_hash_str` |
| `sglang.srt.mem_cache.memory_pool_host` | `MLATokenToKVPoolHost.get_page_buffer_meta` —— **作为对照基准** |

**⇒ 流水线拓扑、线程数、队列交接语义、分片算术、`increment`/`terminate` 语义、收割判定逻辑 —— 全部是 sglang 真实代码在跑。**

### 2.2 替身（只有 4 处）

| # | 真实对象 | 替身 | 替身做了什么 | 保真度验证 |
|---|---|---|---|---|
| 1 | `mooncake.store.MooncakeDistributedStore`（C++ pybind） | `RecordingStore` / `TimingMooncakeStore` | 实现 `batch_is_exist(keys) -> List[int]`、`batch_get_into(keys, ptrs, sizes) -> List[int]`；延迟用 `time.sleep()` 注入；记录调用次数 / 每次 key 数 / 调用线程名 | **V2 / V3 / V4** |
| 2 | `MLATokenToKVPoolHost`（host 内存池） | `StubHostKVCache` | `get_page_buffer_meta` 的 page_first_direct 分支**逐行照抄** `memory_pool_host.py:987`；加上 `alloc` / `free` | **V1（与真实方法逐指针、逐 size 比对）** |
| 3 | `HiRadixCache` 实例状态 | `object.__new__(HiRadixCache)` + 桩字段 | 真实方法用 `Cls.method.__get__(stub)` 绑定；只桩掉 `_insert_helper_host`（返回 0）和 `evict_host` | V5 |
| 4 | 真实 radix tree 的 host 索引写入 | `_insert_helper_host` 返回 0 | 不校验插入结果（插入正确性不在本次结论范围内） | — |

### 2.3 关键技巧：跳 `__init__` 但保留真实方法体

`HiRadixCache.__init__` 需要完整的 device KV pool / server_args / 分布式进程组，本机起不来。做法：

```python
tree = object.__new__(HiRadixCache)          # 跳过 __init__
tree.check_prefetch_progress = HiRadixCache.check_prefetch_progress.__get__(tree)
```

**⇒ 方法体是真的，只有 `self` 上的字段是桩。** 这个技巧让「收割时机决定收益」这类**纯控制流语义**的结论可以在无 GPU、无模型的环境下被真实验证。

---

## 3. 关键区分：哪些结论是模型无关的

这是整套分析的地基。**不区分这两类，就等于把建模当实测。**

### 3.1 模型无关（代码决定，可直接引用）

| # | 结论 | 怎么得到的 |
|---|---|---|
| 1 | 线程拓扑：3 个后台线程；`batch_get_into` 跑在 `prefetch_io_aux_thread` | V4 断言 |
| 2 | RPC 调用次数：`exists` 与 `get` 各 N 次，N 由 `page_size` / `storage_batch_size` 决定 | V2 / V3 断言（真实方法实际发出的调用 vs 算术公式） |
| 3 | 每次 RPC 的 key 数：MLA 1 key/页、MHA 2 key/页 | V1 / V2 |
| 4 | 流水线每级只有 1 线程 → 阶段内 head-of-line blocking | 代码结构 |
| 5 | 收割点的位置（`scheduler.py:2086`）+ 两道 batch 门（`:1993` early return、`:2081` break） | 代码结构 |
| 6 | `completed_tokens` 冻结语义（terminate 后 `increment()` 返回 False） | V5 断言 |
| 7 | 收割时机决定收益（早收割 → 0 token） | V5 断言 |
| 8 | **lat=0 时的纯 Python/结构开销 ≈ 3.2 ms/op** | 实测，且与网络无关 |

### 3.2 模型依赖（取决于我假设的 RPC 延迟）

**所有带 ms 的绝对耗时**：1.45 s、3.2~5.6 ms、0.91 s、5.56 s …

这些数字 = `N × (假设的 lat + 实测的 Python 开销)`。主文档 §3.4 给了完整的敏感性扫描表，把假设换成你线上的真实 RTT 即可：

```
你的真实 RTT ≈ 0.5ms  →  256 请求全部完成 ≈ 0.91 s
你的真实 RTT ≈ 2ms    →  256 请求全部完成 ≈ 1.21 s
你的真实 RTT ≈ 5ms    →  256 请求全部完成 ≈ 1.81 s
```

### 3.3 完全未验证（必须真实环境才能测）

- mooncake master / segment 的真实行为、真实 RTT、续租语义
- 合并 exists+get 后对象是否会被提前回收
- GPU / CUDA 侧：`register_buffer` 失败的影响、零拷贝 buffer 的真实写入
- TP>1 的集合通信一致性（各 rank 收割数量不同时会静默算错 + 挂死）
- 真实 radix tree 插入后的实际命中率
- host 内存池的真实 alloc / evict 压力

---

## 4. 复现步骤

```bash
# 环境（WSL）
cd /home/mzygd/workspace/sglang
.venv/bin/python -V                                  # Python 3.12
.venv/bin/python -c "import torch; print(torch.__version__)"   # 2.9.1+cu128
.venv/bin/python -c "import mooncake"                # ModuleNotFoundError —— 脚本不需要它

# ① 替身保真度自检（**先跑这个**）
.venv/bin/python probes/verify_stub_fidelity.py

# ② 流水线排队实测 + 延迟敏感性扫描（主文档 §3.1 / §3.4 的数据）
.venv/bin/python probes/prefetch_queue_probe.py

# ③ 线程栈抓取（主文档 §2.2 的 backtrace）
.venv/bin/python probes/mooncake_backtrace_repro.py

# ④ MLA RPC 次数表 + 收割时机实验 + 并发场景
.venv/bin/python probes/prefetch_mla_probe.py
```

> 脚本从仓库路径跑即可（`probes/` 已在仓库里）；也可以从任意目录用绝对路径调用。
> ① 的退出码 0 表示替身与真实 sglang 实现一致。

---

## 5. 保真度自检覆盖什么

`probes/verify_stub_fidelity.py` 把「替身是否忠实」变成可执行的断言：

| 检查 | 内容 | 断言 |
|---|---|---|
| **V1** | `get_page_buffer_meta` 替身 vs **真实** `MLATokenToKVPoolHost.get_page_buffer_meta` | 指针个数一致 / **每个指针地址逐一相等** / element_size 列表相等 / MLA 每页恰好 1 个 ptr |
| **V2** | 真实 `_storage_hit_query` 实际发出的 `batch_exists` | 调用次数 == 算术公式 / 每次 key 数 == 分片内页数 / 命中页数 == tokens÷page_size（page_size=64 与 =1 各跑一遍） |
| **V3** | 真实 `_page_transfer` 实际发出的 `batch_get_v1` | 调用次数 == 算术公式 / 每次 key 数 == 页数 |
| **V4** | 线程拓扑 | `batch_get_into` 确实被调用 / 调用线程不是 MainThread / **调用线程名 == `prefetch_io_aux`** |
| **V5** | 收割时机语义 | best_effort 下「收割早于 IO 完成 ⇒ 交付 0 token」/「收割晚于 IO 完成 ⇒ 交付全部 token」 |

### 实测输出（全部 PASS）

```
V1  host 内存池 get_page_buffer_meta 替身 vs 真实 MLATokenToKVPoolHost
  [PASS] 指针个数一致  — real=5 stub=5
  [PASS] 每个指针地址逐一相等
  [PASS] element_size 列表相等  — real=[16384] stub=[16384]
  [PASS] MLA 每页恰好 1 个 buffer 指针  — 5 ptrs for 320 indices (page_size=64)

V2/V3  分片算术 vs 真实方法实际发出的调用  (page_size=64, tokens=8192)
  [PASS] 真实 `_storage_hit_query` 发出的 batch_exists 次数 == 算术公式  — real=1 formula=1
  [PASS] 每次 batch_exists 的 key 数 == 分片内的页数  — real=[128] expect_each=128
  [PASS] 命中页数 == tokens/page_size  — 128 pages for 8192 tokens
  [PASS] 真实 `_page_transfer` 发出的 batch_get_v1 次数 == 算术公式  — real=1 formula=1
  [PASS] 每次 batch_get_v1 的 key 数 == 页数(MLA 每页 1 key)  — real=[128] expect_each=128

V2/V3  分片算术 vs 真实方法实际发出的调用  (page_size=1, tokens=4096)
  [PASS] 真实 `_storage_hit_query` 发出的 batch_exists 次数 == 算术公式  — real=32 formula=32
  [PASS] 每次 batch_exists 的 key 数 == 分片内的页数  — real=[128]×32 expect_each=128
  [PASS] 命中页数 == tokens/page_size  — 4096 pages for 4096 tokens
  [PASS] 真实 `_page_transfer` 发出的 batch_get_v1 次数 == 算术公式  — real=32 formula=32
  [PASS] 每次 batch_get_v1 的 key 数 == 页数(MLA 每页 1 key)  — real=[128]×32 expect_each=128

V4  线程拓扑：batch_get_into 的调用线程
  [PASS] batch_get_into 确实被调用了
  [PASS] 调用线程不是主线程
  [PASS] 调用线程是 prefetch_io_aux_thread

V5  收割时机语义 (best_effort)
  [PASS] 收割早于 IO 完成 ⇒ 交付 0 token  — completed_tokens=0
  [PASS] 收割晚于 IO 完成 ⇒ 交付全部 token  — completed_tokens=256 (expect 256)

ALL CHECKS PASSED
```

**⇒ V2/V3 用的是 sglang 的真实方法体，只是把 mooncake 的返回值换成记录器 —— 所以「RPC 次数」「每次 key 数」是真实实现跑出来的，不是公式推出来的。**

---

## 6. 排队实测的复现性

同一脚本连跑 3 次（256 并发那一行）：

| 轮次 | 全部完成 | 排队等待(末) | `_page_transfer` 均值 |
|---|---|---|---|
| 第 1 次 | 1453.0 ms | 1448.3 ms | 5.59 ms |
| 第 2 次 | 1315.5 ms | 1311.6 ms | 5.08 ms |
| 第 3 次 | 1087.0 ms | 1081.4 ms | 4.19 ms |

**⇒ 同一台机器上就有 ±25% 的运行间波动。绝对值只能当量级看。**

对应的延迟敏感性扫描（lat=0 那档是模型无关的下界）：

| 假设 exists=lat, get=lat | 全部完成 | 每 op 占用 |
|---|---|---|
| lat = 0.00 ms | 824.7 ms | 3.221 ms |
| lat = 2.00 ms | 1211.0 ms | 4.731 ms |
| lat = 20.00 ms | 5559.5 ms | 21.717 ms |

**拟合：每 op 占用 ≈ lat + 2.7~3.2 ms。**

---

## 7. 要做成端到端真实验证，还需要什么

1. **装 mooncake**：`pip install mooncake`，或从 `D:\Codes\Mooncake` 构建
2. **起 mooncake master + segment**
3. **起 sglang**：`--hicache-storage-backend mooncake --hicache-storage-prefetch-policy timeout`
4. **打上主文档 §10 的 5 个点**
5. **跑真实压测**，把真实的 `batch_is_exist` / `batch_get_into` 延迟填回 §3.4 的扫描表
6. **重点验证两件事**：
   - `prefetch_queue.qsize()` / `prefetch_buffer.qsize()` 是否持续增长（流水线饱和）
   - harvest 的 `age`（`now - operation.start_time`）是否远大于真实 IO 耗时（收割被 gate）

---

## 8. 常见质疑与回应

| 质疑 | 回应 |
|---|---|
| 没装 mooncake 你测的什么？ | 测 **sglang 的控制流**。mooncake 只作为「调用契约」被替身实现（返回什么形状、几个 key、什么延迟） |
| 没有模型哪来的请求？ | 不需要模型。`PrefetchOperation` 是普通对象，`HiCacheController` 的方法可以直接驱动 |
| 1.45 s 是不是编的？ | 有原始日志，连跑 3 次 1453 / 1315 / 1087 ms。§3.4 的扫描表显示绝对值随假设线性缩放；**lat=0 时仍有 824.7 ms 是硬事实** |
| 替身会不会掩盖真实行为？ | V1 逐指针比对真实实现；V2/V3 用真实方法体断言实际发出的调用与算术公式一致；V4 断言线程名 |
| 为什么不直接 py-spy 抓线上？ | 本机没有运行中的 sglang 进程；而且要验证的是**逻辑**（谁调谁、几次调用、什么时候收割），不是性能热点 |
| 为什么不装 mooncake 直接测？ | 本机 GPU 是 RTX 3070 (sm_86)，跑不了线上那种 MLA 模型；装 mooncake 也要先起 master + segment，投入产出比不如先把逻辑钉死 |

# read-sglang

sglang 源码阅读笔记 / 线上问题排查记录。

分析环境：WSL `/home/mzygd/workspace/sglang`（v0.5.9, detached HEAD @ `bbe9c7ee`），
Python 3.12 / torch 2.9.1+cu128。
配套 Mooncake C++ 源码见 `D:\Codes\Mooncake`（= WSL `/mnt/d/Codes/Mooncake`）。

---

## 文档索引

| 文档 | 主题 | 状态 |
|---|---|---|
| [sglang-hicache-mooncake-prefetch.md](./sglang-hicache-mooncake-prefetch.md) | **HiCache + Mooncake prefetch 性能分析**：调用 mooncake 的线程与 backtrace、PrefetchOperation 生命周期耗时分解、scheduler 主循环与收割点（harvest）被 `batch_is_full` 卡住的机制、mooncake master 侧 `batch_is_exist` 带写续租 vs `batch_get_into` 纯读的差异、MLA vs MHA 的 RPC 次数差异、已知问题清单与分级修复方案 | ✅ |
| [TESTING.md](./TESTING.md) | **测试方法论**：本机没有 mooncake / 没有模型的情况下怎么测、哪些是真实代码哪些是替身、替身保真度自检、**哪些结论模型无关 / 哪些是建模推导 / 哪些完全未验证** | ✅ |
| [`probes/`](./probes) | 6 个探针脚本，**可独立复现每一个数据点**；含 `verify_stub_fidelity.py`（替身保真度自检） | ✅ |

---

## 当前分析的主要结论（速览）

**问题现象**：`_page_transfer` 很快，但整个 `PrefetchOperation` 的耗时很高。

1. **调用 mooncake `batch_get_v1` 的是后台线程 `prefetch_io_aux_thread`**，不是 scheduler 主线程。
2. **MLA 下 `batch_is_exist` 与 `batch_get_into` 的 RPC 次数严格 1:1**，单次 exists 更便宜 → IO 本身不是瓶颈。
3. **真正的瓶颈是排队**：`prefetch_thread` / `prefetch_io_aux_thread` 各只有 1 个线程 → 阶段内 head-of-line blocking。**模型无关的硬事实：即使把 mooncake 延迟设为 0，256 个请求的 prefetch 仍需 824.7 ms 才全部完成**（每 op 3.22 ms 纯 Python/结构开销）。在假设 RPC 延迟 2 ms 的模型下，排队等待涨到 **1.2~1.5 s**，而 `_page_transfer` 自身恒定 **3.2~5.6 ms**（绝对值随假设线性缩放，见文档 §3.4）。
4. **收割（harvest）被 `batch_is_full` 卡住**：唯一收割点 `scheduler.py:2086` 位于 staging loop 内，而该 loop 在 batch 满时会 `break`（`:2081`）或整个 early-return（`:1993`）→ **底层 IO 早完成了，radix tree 索引却迟迟建不上** → L3 收益为 0，且 host 页与 `prefetch_tokens_occupied` 一直被占着。
5. **收割周期 ≈ 一次 forward 的墙钟 T**：overlap 调度器里 scheduler 线程阻塞在 `pop_and_process()` → `copy_done.synchronize()`（等上一批结果拷回 CPU），不是空轮询。
6. **`prefetch_tokens_occupied` 只在收割/abort 时递减**，而 `SGLANG_REQ_WAITING_TIMEOUT` 默认 -1 不启用 → 收割不来 → 占用不释放 → `prefetch_rate_limited()` 触发 → **整套 prefetch 机制停摆**。
7. mooncake 侧：`batch_is_exist` 和 `batch_get_into` **都打 master**；差别是 Exist **每个 key 都要写 metadata 续租**，GetReplicaList 是纯读 → 「exists 只是查在不在所以应该更快」在实现层不成立。
8. **MLA（page_size=64）1 次 RPC / 8192 token；MHA（page_size=1，默认）会把 RPC 次数放大 64 倍。**

详见文档正文，含完整实测数据、行号索引与分级修复方案（P0~P5）。

---

## 复现方式

全部实测数据由 [`probes/`](./probes) 下的 6 个脚本产出，**每个数据点都可独立复现**。

```bash
# 环境：WSL / sglang v0.5.9（脚本不需要 mooncake，也不需要 GPU）
cd /home/mzygd/workspace/sglang

# ① 替身保真度自检（先跑这个；退出码 0 = 替身与真实 sglang 实现一致）
.venv/bin/python <仓库路径>/probes/verify_stub_fidelity.py

# ② 流水线排队实测 + 延迟敏感性扫描
.venv/bin/python <仓库路径>/probes/prefetch_queue_probe.py

# ③ 线程栈抓取（faulthandler backtrace）
.venv/bin/python <仓库路径>/probes/mooncake_backtrace_repro.py

# ④ MLA RPC 次数表 + 收割时机实验
.venv/bin/python <仓库路径>/probes/prefetch_mla_probe.py
```

**测试的边界（务必先读 [TESTING.md](./TESTING.md)）：**

- 真实跑的是 **sglang 的代码路径**（`HiCacheController` / `PrefetchOperation` / `MooncakeStore` / `HiRadixCache` 全部是 import 自已安装的 sglang，一行没改）
- 替换掉的只有 4 处：mooncake 的 C++ pybind 对象、host 内存池、radix tree 状态、radix 插入
- **模型无关的结论**（可直接引用）：线程拓扑、RPC 次数与每次 key 数、流水线结构、收割点位置与两道 batch 门、`completed_tokens` 冻结语义、lat=0 时的纯 Python 开销
- **建模推导的结论**（所有带 ms 的绝对耗时）：取决于脚本里假设的 RPC 延迟 → 主文档 §3.4 有完整敏感性扫描表
- **完全未验证**：mooncake 真实行为与延迟、GPU/零拷贝、TP>1 集合通信、真实 radix 插入命中率

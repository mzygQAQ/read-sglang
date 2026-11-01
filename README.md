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

---

## 当前分析的主要结论（速览）

**问题现象**：`_page_transfer` 很快，但整个 `PrefetchOperation` 的耗时很高。

1. **调用 mooncake `batch_get_v1` 的是后台线程 `prefetch_io_aux_thread`**，不是 scheduler 主线程。
2. **MLA 下 `batch_is_exist` 与 `batch_get_into` 的 RPC 次数严格 1:1**，单次 exists 更便宜 → IO 本身不是瓶颈。
3. **真正的瓶颈是排队**：`prefetch_thread` / `prefetch_io_aux_thread` 各只有 1 个线程、两级串行 → head-of-line blocking。实测 256 并发时排队等待涨到 **1.45 秒**，而 `_page_transfer` 自身恒定 **3.6~5.6 ms**。
4. **收割（harvest）被 `batch_is_full` 卡住**：唯一收割点 `scheduler.py:2086` 位于 staging loop 内，而该 loop 在 batch 满时会 `break`（`:2081`）或整个 early-return（`:1993`）→ **底层 IO 早完成了，radix tree 索引却迟迟建不上** → L3 收益为 0，且 host 页与 `prefetch_tokens_occupied` 一直被占着。
5. **收割周期 ≈ 一次 forward 的墙钟 T**：overlap 调度器里 scheduler 线程阻塞在 `pop_and_process()` → `copy_done.synchronize()`（等上一批结果拷回 CPU），不是空轮询。
6. **`prefetch_tokens_occupied` 只在收割/abort 时递减**，而 `SGLANG_REQ_WAITING_TIMEOUT` 默认 -1 不启用 → 收割不来 → 占用不释放 → `prefetch_rate_limited()` 触发 → **整套 prefetch 机制停摆**。
7. mooncake 侧：`batch_is_exist` 和 `batch_get_into` **都打 master**；差别是 Exist **每个 key 都要写 metadata 续租**，GetReplicaList 是纯读 → 「exists 只是查在不在所以应该更快」在实现层不成立。
8. **MLA（page_size=64）1 次 RPC / 8192 token；MHA（page_size=1，默认）会把 RPC 次数放大 64 倍。**

详见文档正文，含完整实测数据、行号索引与分级修复方案（P0~P5）。

---

## 复现方式

本文档的实测数据由一组探针脚本产出（真实 sglang 代码路径 + 可控延迟的 mooncake 替身）。
脚本目前保留在本地，未随仓库提交；核心复现命令形如：

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/mzygd/workspace/sglang && .venv/bin/python <脚本路径>"
```

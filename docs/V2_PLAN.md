# v2 计划：把 CNN 训练搬到显存上（设备常驻）

> 草案 v0.1 · 2026-10-28
> 前置阅读：`README.md`（项目总览）、`BASELINE.md`（v1.0 GPU GEMM 基线）、`docs/worklog/`（开发日志）

## 0. 为什么做 v2

v1 已经跑通"C++ 从零实现 + CUDA 加速"，但实测收益有限：

| 指标 | v1 实测（batch=2000，10 iter/epoch） |
|---|---|
| CPU 全 CPU | 12.9 s/epoch |
| v1.1 自适应（卷积自动上 GPU） | 10.2 s/epoch（约 **1.27×**） |
| 卷积 kernel 有效算力 | 0.6 ~ 4.7 TFLOPS（好的层已达 4096³ SGEMM 的 85%） |
| 卷积在 epoch 中的占比 | 约 31% |
| 每次卷积调用的边界成本 | pack/unpack/H2D/D2H 合计 8 ~ 49 ms，而 kernel 仅 0.4 ~ 3.4 ms |

结论：**kernel 不慢，慢在数据怎么进出 GPU，以及大量算子仍留在 CPU。**
v2 的目标是把整条 CNN 前向/反传链路变成"一次上传、显存内跑完、一次回传"，用数据回答一个问题：

> 同一台机器（RTX 5060 Laptop / i7-14650HX），把边界成本压掉之后，这个网络能快多少？

## 1. 验收指标（Definition of Done）

1. **数值**：`NN_ACCEL=cpu` 与 `NN_ACCEL=gpu` 逐元素对拍，最大相对误差 < 1e-4（FP32 正常范围）。
2. **收敛**：固定随机种子，2 epoch（10×2000 样本）后测试集准确率不低于 CPU 版 2 个百分点。
3. **性能**：epoch 时间 ≤ 4 s（目标 3~6×，对比 v1 CPU 12.9 s）。
4. **可复现**：一条命令跑完"构建 → 训练 → 基准"，结果写入 `docs/bench/`。

## 2. 现状与瓶颈（来自 v1.1 剖析）

每批 2000 样本的 CPU 时间分解（估算）：

```
CPU 版每批 ≈ 1.29 s
├─ 卷积（9 次调用）                    ≈ 0.40 s   ← v1.1 已把大头交给 GPU
├─ MLP（矩阵乘）                       ≈ 0.05 s
├─ 周边算子（padding/dilate/transpose/
│   hadamard/pool/activation/flatten）  ≈ 0.60 s   ← 全量 CPU 拷贝，v2 的主要目标
└─ 数据准备/调度                        ≈ 0.24 s
```

v1.1 GPU 路径的单次调用明细（稳态）：

| 调用 | kernel | pack | H2D | D2H | unpack | 合计 |
|---|---|---|---|---|---|---|
| 前向层 0 | 0.56 ms | 1.5 | 0.4 | 3.2 | 16.5 | 22.3 ms |
| 前向层 1 | 0.40 | 2.6 | 0.8 | 1.2 | 3.3 | 8.5 ms |
| 反传层 1 dA | 0.43 | 15.5 | 0.8 | 0.9 | 13.7 | 31.9 ms |
| 反传层 0 dA | 3.38 | 34.5 | 5.7 | 0.4 | 4.5 | 48.7 ms |
| 反传 dK ×3 | —— | —— | —— | —— | —— | 回退 CPU（并行度太低） |

两个直接结论：
1. **pack/unpack（double↔float 转换）是最大单项**，且其中一半时间在 `double` 存储上；
2. **周边算子全是 CPU 整批拷贝**，在设备常驻后可以直接消灭。

## 3. 目标架构

### 3.1 三层结构

| 层 | 职责 | 与 v1 的关系 |
|---|---|---|
| **L0 数学库** | `Matrix_Double` / `Batch_Matrix` / `Rational` 等 | **保留不动**，作为 CPU 参考实现 |
| **L1 设备张量 `GpuTensor`** | float 连续显存；`[R][C][H][W]` 布局；RAII 管理；只在边界做 H2D/D2H | 新增 |
| **L2 网络执行器** | 层算子（conv/pool/act/fc）、每层持有 device 缓冲、前向/反传调度 | 重构（从 `NeuralNetwork_MULCNN` 迁移） |

原则：**v1 的 CPU 路径不删**，保留为对拍参考与回退（`NN_ACCEL=cpu`）。

### 3.2 布局约定

```
输入  x : [R][C_in ][H ][W ]  float
输出  y : [R][C_out][Ho][Wo]  float
权重  w : [C_out][C_in][K][K]  float     （Ho=(H+2p-K)/s+1）
偏置  b : [C_out]
```

- 所有活跃张量常驻显存；**只在 epoch 开始（上传数据）和结束（回传 logits）时跨 PCIe**。
- 核内继续用 tile + shared memory；分块参数由 host 端按"计算量 + 线程数 + 共享内存"自动选（见 v1.1 `GetConvGpuThreadCount`）。

### 3.3 算子清单

前向：
- [ ] `conv_forward`：im2col 物化（第一版）或 implicit-GEMM（第二版）
- [ ] `bias + activation` 融合进 conv 输出（省一次全量读写）
- [ ] `maxpool_forward`：同时保存 argmax 索引
- [ ] `flatten`

反传：
- [ ] `conv_backward_dA` / `conv_backward_dK`
- [ ] `activation_backward`
- [ ] `maxpool_backward`（用前向保存的索引）
- [ ] FC（MLP）：第一版留在 CPU，v2.1 再上 GPU

## 4. 里程碑

| 阶段 | 内容 | 预计 | 交付/验收 |
|---|---|---|---|
| **M0 收尾** | v1 提交、tag、README 初版、分支整理 | 0.5~1 天 | 仓库干净、能一键跑基准 |
| **M1 边界优化** | pack/unpack 改 AVX2；pinned 锁页；矩阵乘 GPU 路径加锁页暂存；记录前后数据 | 3~5 天 | 卷积边界成本减半（130→60 ms/批），epoch ≤ 9.5 s |
| **M2 设备常驻前向** | `GpuTensor`；conv/bias/act/pool 融合 kernel；CPU↔GPU 只两次 | 5~7 天 | 前向对拍 < 1e-4；前向耗时 < 1 s/epoch |
| **M3 反传设备化** | dA/dK/pool-backward；训练收敛 | 5~7 天 | 验收指标 1/2 达标；epoch ≤ 4 s |
| **M4 文档与展示** | README 定稿、两篇技术博客、基准表 | 3~5 天 | 可发给老师/公开 |

**止损点**：M2 结束时如果设备常驻的复杂度超出预期（例如对拍无法收敛），回退到 M1 状态（分支 + tag 都在），把 M1 的收益和 v1 的分析写成博客即可——那本身也是可交付成果。

## 5. 明确不做（防止 scope 蔓延）

- 不做通用 DL 框架：无 autograd、无任意计算图、无多设备。
- 不追求 cuDNN 级别性能：目标是**可解释的 3~6×**，每一步都要有测量依据。
- 不重写数学库、不重写 SFML 应用层。
- 不做混合精度/量化/分布式。

## 6. 目录结构（建议）

```
src/math/      Matrix_*.cpp, Rational, Polynomial, MathFunction      （现有，不动）
src/nn_cpu/    NeuralNetwork_*.cpp, Batch_Matrix.cpp                （现有，作为参考）
src/nn_gpu/    gpu_tensor.cu/h, conv.cu, pool.cu, act.cu, fc.cu     （新增）
src/apps/      main_bench.cpp（基准）, main_demo.cpp（演示）          （从 main.cpp 拆出）
tests/         conv_test.cpp, fc_test.cpp, grad_check.cpp           （新增）
docs/          V2_PLAN.md, worklog/, bench/                         （已有 worklog）
```

## 7. 对拍与测试

1. 每个 GPU kernel 必须有 CPU 参考实现 + 相对误差断言（`tests/`）；
2. 小网络做**梯度检查**：有限差分 vs 反向传播；
3. 固定 seed 的训练可复现性；
4. 性能数据必须记录：日期、硬件、形状、前后对比（写入 `docs/bench/`）。

## 8. 记录规范（devlog 模板）

```markdown
## YYYY-MM-DD 目标
- 假设：
- 改动：
- 数据（前后对比）：
- 结论：
- 下一步：
```

要求：**每次性能改动必须附数据**；失败实验也要记（"为什么这个方向不行"是最有价值的部分）。

## 9. 风险与回退

| 风险 | 应对 |
|---|---|
| 时间不够（课业重） | 按里程碑推进，每个里程碑都可独立交付 |
| FP32 精度影响收敛 | 与 CPU 版对拍 + 固定 seed 比较；必要时关键层保留 double |
| 设备常驻调试困难 | 每个 kernel 单独对拍；保留 `NN_ACCEL=cpu` 回退 |
| 共享内存/占用率问题 | 沿用 v1.1 的"阈值 + 线程数 + shared 预算"三判据 |

## 10. 时间预算

- 每周 6~8 小时，总计 4~6 周；
- 优先 M0 → M1（低风险、确定收益），再决定 M2 的深度；
- 文档与 devlog 同步写，不放到最后补。

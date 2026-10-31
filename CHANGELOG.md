# 更新日志

版本约定：
- 每个 `v1.x` 都是**可构建、可运行、有基准数据**的快照；
- `main` 分支保持可构建；大改动走独立分支（如 `v2-dev`）；
- 每次性能改动必须在 `docs/worklog/` 留下"目标 / 假设 / 数据 / 结论"。

## 未发布

- **v1.2（计划）**：pack/unpack 改 AVX2 + 锁页/异步传输 → 目标：卷积边界成本减半（约 130 ms → 60~80 ms/批），epoch ≤ 9.5 s。
- **v2.0（计划）**：设备常驻 CNN 执行模型（`GpuTensor` + 融合 kernel + 前后向全程显存），目标 epoch ≤ 4 s。详见 [docs/V2_PLAN.md](docs/V2_PLAN.md)。

## 未发布的工程改动（演示/可移植性）

- 完整工程 SFML 路径改为可配置：默认 `third_party/SFML-3.1.0`，可用 `SFML.local.props`（不入库）或环境变量 `SFML_DIR` 覆盖；
- `Demo_CPU` 输出末尾增加最终准确率/总耗时总结；
- `Bench` 在没有 CUDA 设备的机器上自动只测 CPU（GPU 模式跳过），无需改代码；

## v1.1（2026-10-28）

- **CPU 卷积**：im2col + 4×8 寄存器分块 + 分块展开，与朴素实现做逐元素对拍（相对误差 ~1e-7）。
- **CUDA 卷积**：手写直接卷积 kernel（共享内存 tile + 8 通道 × 2×2 位置 = 32 个寄存器累加器）；
  修复 z 维分块索引、共享内存申请量、写回坐标等 bug；与 CPU 版对拍通过。
- **GPU 矩阵乘**：新增"直接写入 out"接口；修复每次调用创建 `cudaEvent` 未销毁导致的事件泄漏。
- **自适应调度**：`Matrix_Double_Mutiple_Auto` / `GetConvolutionMatrixMul_Auto`，
  按"计算量阈值 + 总线程数"选择 CPU/GPU；反传 dK 的瘦形状自动回退 CPU。
- **训练基准**：`main` 改为可配置基准（`NN_ACCEL` / `NN_EPOCHS` / `NN_ITERS` / `NN_BATCH`），
  实测卷积 GPU 化后 epoch 12.9 s → 10.2 s（≈1.27×，瓶颈分析见 README）。
- **文档**：README 重写、`docs/V2_PLAN.md`、`docs/PITFALLS.md`、开发日志目录。

## v1.0（2026-10-07）

- 手写 CUDA SGEMM：共享内存 tiling + 4×4 寄存器分块 + 合并访存 + 设备缓冲常驻；
  4096³ 完整 kernel 25.09 ms（5.48 TFLOPS），端到端 ≈ 72.9 ms。
- `ABLATION_MODE` 消融实验（只加载 / 只计算 / 完整）；
  用"工作集缩到 L2 后测速不变"证伪 DRAM 带宽假设，定位计算阶段瓶颈为 LDS 指令条数。
- `README.md` / `BASELINE.md` 初版；CPU GEMM 4×8 分块达 37.3 GFLOPS。

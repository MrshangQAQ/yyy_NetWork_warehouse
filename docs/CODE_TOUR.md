# 代码导览（给读代码的同学）

> 目标：让你在 **15 分钟**内看懂这个项目在做什么、该从哪读起。
> 先看这份导览，再按第 2 节的顺序读代码，基本不会迷路。

## 0. 一句话

**从零实现的 C++ 数学库 + 神经网络 + CUDA 算子**：
矩阵/卷积/池化/反向传播全部手写，没有 Eigen、BLAS、cuBLAS、cuDNN、PyTorch。

- "从零"的含义：只用 C++ 标准库 + CUDA Runtime（SFML 仅用于手写数字画板 Demo）；
- 定位：学习项目 + 性能工程实验记录，不是生产级框架；
- 每个性能结论都有对应的测量数据（见 `README.md`、`BASELINE.md`、`docs/`）。

## 1. 想先跑起来看效果？

不用 CUDA / SFML / 数据集：

- 打开 **`Demo_CPU.vcxproj`** → `Release | x64` → `Ctrl+F5`
- 约 1 秒看到：精确有理数、CPU GEMM 基准、MLP 训 XOR、CNN 训条形图分类
- 详细说明：[docs/DEMO_CPU.md](DEMO_CPU.md)

## 2. 推荐阅读顺序（从依赖底层往上）

| 顺序 | 文件 | 看点 |
|---|---|---|
| 1 | `Matrix.h` / `Matrix_Double.cpp` | 行主序连续存储；OpenMP + AVX2 + 4×8 寄存器分块的 CPU GEMM；有理数/多项式/矩阵分解等数学库 |
| 2 | `Batch_Matrix.h` / `Batch_Matrix.cpp` | 4 维张量 `[R][C][H][W]` 的存储约定；im2col + GEMM 的 CPU 卷积；maxpool 的 argmax 索引缓存 |
| 3 | `NeuralNetwork.h` / `NeuralNetwork_MLP.cpp` / `NeuralNetwork_MULCNN.cpp` | 手写前向/反向：ReLU/Sigmoid/Tanh、反池化（按 argmax scatter）、卷积转置、dK 归约；`Complex_NeuralNetwork` 负责串层 |
| 4 | `Loss.h` / `Loss.cpp` / `APP_MNIST.cpp` | Softmax + 交叉熵梯度；MNIST/Fashion-MNIST 的 IDX 文件读取 |
| 5 | `Matrix_Double_Mutiple_GPU.cu` | 手写 CUDA SGEMM：共享内存 tiling + 4×4 寄存器分块 + 合并访存；`ABLATION_MODE` 消融实验（只加载/只计算/完整） |
| 6 | `imcol2_GPU.cu` | 手写直接卷积 kernel：共享内存 tile + 32 个寄存器累加器；CPU/GPU 自适应调度（阈值 + 总线程数判据） |
| 7 | `Bench.cpp` / `run_bench.cmd` / `run_bench.ps1` | 单算子微基准：同进程交替采样、min/median、有效 GFLOP/s；这是判断"优化有没有用"的尺子 |
| 8 | `README.md` / `BASELINE.md` / `docs/` | 实测数据、瓶颈分析（LDS 指令墙、GPU 卷积为何只有 1.27×）、踩坑记录、v2 计划 |

## 3. 读代码时值得注意的设计决策

1. **主机 double、GPU float**：消费级 GPU 的 FP64 只有 FP32 的 1/64，所以只在主机↔设备边界做一次转换；
2. **CPU 卷积 = im2col + GEMM**，**GPU 卷积 = 直接卷积**：前者复用寄存器分块 GEMM，后者省掉 im2col 的物化和读放大；
3. **GPU 自适应调度**：按"计算量阈值 + 总线程数 + 共享内存预算"选 CPU/GPU，反传 dK 这类瘦长形状自动回退 CPU；
4. **边界优化**：矩阵乘的 A 路径 = 主机侧 `double→float`（AVX2/OpenMP）→ 锁页缓冲 → 异步 H2D → float GEMM → 异步 D2H → 主机侧转回 double（详见 `Matrix_Double_Mutiple_GPU.cu` 顶部注释）；
5. **测量方法**：epoch 级时间波动可达 ±60%，所以优化决策用 `Bench` 的单算子数据（波动 ±1~5%）。

## 4. 文件规模速查（大致）

| 模块 | 文件 | 规模 |
|---|---|---|
| 数学库 | `Matrix.h` + `Matrix_*.cpp` | ~2000 行 |
| 张量/卷积 | `Batch_Matrix.h/.cpp` | ~1000 行 |
| 神经网络 | `NeuralNetwork_*.cpp` | ~800 行 |
| CUDA | `*.cu` | ~1200 行 |
| 测量/应用 | `Bench.cpp` / `main.cpp` / `APP_MNIST.cpp` | ~600 行 |
| 文档 | `README.md` / `BASELINE.md` / `docs/` | ~3000 行 |

## 5. 完整工程的环境要求

| 依赖 | 要求 |
|---|---|
| Visual Studio | 2022 或 2026，勾选"使用 C++ 的桌面开发" |
| CUDA Toolkit | ≥ 12.8（目标 `sm_120`；更老的显卡需改工程里的 `CodeGeneration`） |
| SFML | 3.1.0（仅手写数字画板 Demo）；解压到 `third_party/SFML-3.1.0`，或用 `SFML.local.props` / 环境变量 `SFML_DIR` 指定路径 |
| 数据集 | 已随仓库提供于 `data/`（Fashion-MNIST，MIT License）；也可换成原始 MNIST |

> 完整工程目前是"能编译、可运行"的状态；`Batch_Matrix_GPU.*` 是 v2 设备常驻的重构中代码，
> 接口可能还会变，**不影响**上面第 1 节推荐的阅读路线。

## 6. 已知限制（如实说明）

- 代码是长期学习项目累积的，**早期与后期风格不统一**（命名、注释、错误处理）；
- 测试以"对拍"为主（卷积 GPU vs CPU、矩阵乘 vs CPU），没有完整的单元测试/CI；
- v2（设备常驻/减少 CPU↔GPU 搬运）还在进行中；
- 完整训练需要 52 MB 数据集，未纳入版本库。

## 7. 如果只想聊三个问题

1. 为什么卷积 GPU 版只比 CPU 快 **1.27×**？（算术强度 + 边界成本 + Amdahl，见 `README.md`）
2. 手写 CUDA GEMM 的瓶颈为什么是 **LDS 指令条数**而不是 DRAM？（见 `BASELINE.md` 的消融实验）
3. 下一步怎么做才能拿到 **3~6×**？（设备常驻，见 `docs/V2_PLAN.md`）

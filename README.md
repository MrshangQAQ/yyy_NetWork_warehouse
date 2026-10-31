# 高等代数 —— 从零实现的 C++ 数学库 / 神经网络 / CUDA 性能实验

> 只用 C++ 标准库 + CUDA Runtime，从零写出矩阵与符号计算 → MLP / 多通道 CNN → CUDA 算子，
> 并对每一次优化给出可复现的测量（包括**没变快**的那些）。

> **只想快速看看效果？** 不需要 NVIDIA 显卡 / CUDA / SFML / 数据集：
> 打开 `Demo_CPU.vcxproj` → 选 `Release | x64` → `Ctrl+F5`，约 1 秒跑完数学库 + MLP + CNN 演示
> （说明见 [docs/DEMO_CPU.md](docs/DEMO_CPU.md)）。
## 这是什么 / 不是什么

**是**：一个数学系学生的自学项目，主线是"从零实现 + 性能工程"。
仓库里既有可运行的 MNIST 格式训练流程，也有每一步为什么这样写的实验记录。

**不是**：生产级深度学习框架。没有自动微分、没有通用计算图、没有分布式，
也没有用 Eigen / BLAS / cuBLAS / cuDNN —— 这是刻意的。

**"从零"的含义**：矩阵运算、卷积、池化、激活、反向传播全部手写；
允许的依赖只有 C++ 标准库、CUDA Runtime（以及仅用于手写数字画板 Demo 的 SFML）。

## 如果你只有 5 分钟

1. 看下面的[实测数据](#实测数据可复现)两张表；
2. 跑一条命令（见[快速开始](#快速开始)），1 分钟内看到 CPU 与 GPU 的 epoch 时间对比；
3. 再看[为什么 GPU 卷积只快了 1.27 倍](#为什么-gpu-卷积只快了-127-倍)——
   这是这个项目里最"研究式"的一段：kernel 本身已经跑到 4.7 TFLOPS，
   瓶颈在数据怎么进出 GPU。

## 快速开始

**环境**：Visual Studio 2022+（C++20）、CUDA Toolkit ≥ 12.8（目标 `sm_120`，RTX 50 系）、SFML 3.1.0（仅手写数字画板 Demo）、Release | x64。

**SFML 路径**：工程默认在仓库内 `third_party/SFML-3.1.0` 查找（该目录不入库）。放在别处时，把 `SFML.local.props.example` 复制为 `SFML.local.props` 并改一行路径即可（该文件被 `.gitignore` 排除），或设置环境变量 `SFML_DIR`。

```powershell
# 1) 构建（.cu 有改动时必须「重新生成解决方案」，详见 docs/PITFALLS.md）
#    用 VS 打开 高等代数.slnx，选 Release | x64 → 重新生成解决方案

# 2) 准备数据：把 MNIST 格式的 4 个 idx 文件放到仓库根目录（已被 .gitignore 排除）：
#    train-images.idx3-ubyte / train-labels.idx1-ubyte
#    t10k-images.idx3-ubyte  / t10k-labels.idx1-ubyte
#    注意：本仓库使用的这批文件内容实际是 Fashion-MNIST（见 PITFALLS 第 5 条）

# 3) 训练基准（默认：卷积自动走 GPU，矩阵乘走 CPU）
cd 高等代数
.\x64\Release\高等代数.exe

> 没有 NVIDIA 显卡时，`NN_BENCH=matmul/conv/all` 会自动只测 CPU（GPU 模式自动跳过）。

# 4) 对比与调参
$env:NN_ACCEL="cpu"     # cpu / conv / gpu / matmul
$env:NN_EPOCHS="2"; $env:NN_ITERS="10"; $env:NN_BATCH="2000"
.\x64\Release\高等代数.exe
```

预期输出：

```
[bench] NN_ACCEL=conv  conv=auto(GPU)  matmul=CPU
[bench] train=60000  test=10000
[bench] epoch 1  time=10890 ms  acc=...
[bench] epoch 2  time=10246 ms  acc=...
```

> 准确率数字只用于确认"还能学"；短基准的主要目的是测时间，不是刷精度。

## 实测数据（可复现）

### CPU GEMM（1024³ × 100 次，i7-14650HX，OpenMP + AVX2 + 4×8 寄存器分块）

| 线程数 | 耗时（100 次） | 算力 |
|---|---|---|
| 1 | 34389 ms | 6.2 GFLOPS |
| 8 | 8386 ms | 25.6 GFLOPS |
| 24 | 5762 ms | 37.3 GFLOPS |

### GPU GEMM（4096³，RTX 5060 Laptop，double→float 边界转换）

| 阶段 | 耗时 |
|---|---|
| H2D | 31.5 ms |
| MODE=1（只加载） | 7.82 ms |
| MODE=2（只计算） | 21.87 ms |
| MODE=0（完整 kernel） | **25.09 ms（5.48 TFLOPS）** |
| D2H | 16.3 ms |
| 端到端 | ≈ 72.9 ms |

用 `ABLATION_MODE` 消融实验把"加载"和"计算"分开计时；进一步用"工作集缩到 L2 后测速不变"
证伪了 DRAM 带宽假设，把计算阶段瓶颈定位到 **LDS 指令条数**（每轮 8 LDS : 16 FMA，
而硬件 LDS:FMA 吞吐为 1:4 ⇒ 结构上限约为峰值的 50%）。

### GPU 卷积 kernel（batch=2000，batch 内直接卷积：shared tile + 32 个寄存器累加器）

| 调用 | 计算量 | kernel 时间 | 有效算力 |
|---|---|---|---|
| 前向层 0 | 2.65 GFLOP | 0.56 ms | **4.7 TFLOPS** |
| 前向层 1 | 1.64 GFLOP | 0.40 ms | 4.1 TFLOPS |
| 反传层 1 dA | 1.64 GFLOP | 0.43 ms | 3.8 TFLOPS |
| 反传层 0 dA | 2.03 GFLOP | 3.38 ms | 0.6 TFLOPS（C_out=1，瘦 N 退化） |

### 训练集成（2 epoch × 10 iter × batch 2000）

| 配置 | epoch 1 | epoch 2 | 相对 CPU |
|---|---|---|---|
| 全 CPU | 12.9 s | 13.8 s | 1× |
| **自适应（卷积 GPU）** | 10.9 s | **10.2 s** | **≈ 1.27×** |

> 机器状态有波动，交替测量 3 轮得到 1.25~1.35×；这里只给保守区间。

## 为什么 GPU 卷积只快了 1.27 倍

反直觉但可解释：

1. **kernel 并不慢**：好的层已经到 4.7 TFLOPS，是 4096³ GEMM（5.48 TFLOPS）的 85%；
2. **每次调用的边界成本是 kernel 的 5~50 倍**：`pack`（double→float，CPU）+ H2D +
   `kernel` + D2H + `unpack`（float→double，CPU），合计 8~49 ms，而 kernel 只有 0.4~3.4 ms；
3. **阿姆达尔**：卷积只占 epoch 的约 31%，其余是 MLP、池化/激活/padding/dilate/transpose/Hadamard
   和数据准备——它们全在 CPU；
4. **算术强度差 30~50 倍**：4096³ 约 340 FLOP/byte，卷积约 6~11 FLOP/byte，
   所以 GPU 在卷积上更像"带宽机"。

完整剖析与 v2 的解法（设备常驻）见 [docs/V2_PLAN.md](docs/V2_PLAN.md)。

## 项目结构

| 模块 | 文件 | 说明 |
|---|---|---|
| 代数基础 | `Rational` / `Complex` | 精确有理数；复数实虚部均为有理数 |
| 多项式 | `Polynomial` | Karatsuba 分治乘法 + FFT 乘法 |
| 矩阵族 | `Matrix_Double` / `Matrix_Complex` / `Matrix_Lambda` | 初等变换、高斯消元、对角化、Smith 标准型、正交分解 |
| 符号计算 | `MathFunction` | 表达式树 + 符号微分 |
| 神经网络 | `NeuralNetwork_MLP` / `NeuralNetwork_MULCNN` | 手写前向/反向；CNN 走 im2col + GEMM |
| CPU 性能 | `Matrix_Double.cpp` / `Batch_Matrix.cpp` | OpenMP + AVX2 + 寄存器分块；卷积分块展开 |
| GPU | `Matrix_Double_Mutiple_GPU.cu` / `imcol2_GPU.cu` | 手写 SGEMM；直接卷积 kernel |
| 调度/基准 | `Matrix_Double_Mutiple_Auto` / `GetConvolutionMatrixMul_Auto` / `main.cpp` | CPU/GPU 自适应选择；训练基准 |

## 关键设计决策

1. **double 存储、GPU 内 FP32 计算**：FP64 只有 FP32 的 1/64（约 0.25 vs 16 TFLOPS），
   在 GPU 上算 double 必输 CPU；因此只在主机↔设备边界做一次转换。
2. **CPU 卷积 = im2col + GEMM**：把卷积变成可复用寄存器分块的矩阵乘；
   并推导了 `bytes/FLOP ≈ 8/C_out` 的判据，用于判断小输出通道时是否该退回朴素卷积。
3. **GPU GEMM：共享内存 tiling + 4×4 寄存器分块 + 合并访存**，
   用消融实验和 bank-conflict 计算解释每次参数调整的得失（含负优化）。
4. **GPU 卷积：直接卷积 + 32 个寄存器累加器**（8 输出通道 × 2×2 位置），
   分块参数由"共享内存预算 + 每 block 线程上限 + 总线程数"三条判据共同决定；
   并行度太低的形状（如反传 dK）自动回退 CPU。
5. **自适应调度**：不是一个"开关"，而是按计算量 + 并行度 + 传输量的启发式；
   实测 GPU 矩阵乘在当前形状（128×1024×2000）下反而不如 CPU，因此在默认配置里关闭。

## 已知问题 / 局限

- 代码是三个月里边学边写的，**早期文件的风格与后期不统一**（命名、错误处理、注释）；
  正在按 [docs/V2_PLAN.md](docs/V2_PLAN.md) 分层重构。
- 测试还不完整：卷积有与朴素实现的对拍（相对误差 ~1e-7），但缺少独立的 `tests/` 工程、
  梯度检查和 CI。
- GPU 卷积训练集成只有 1.27×：边界成本与 Amdahl 限制，详见上文。
- 反传 dK 的畸形形状（输出只有 9×9）并行度太低，目前走 CPU。
- 完整训练（30 epoch）的精度记录在 [docs/worklog/](docs/worklog/)，但尚未整理成可复现脚本。
- 数据集文件不入库（约 55 MB）；本仓库这批文件实际是 Fashion-MNIST。

## 开发过程（为什么代码会长成这样）

- 大一：C 语言课 → 对计算机产生兴趣 → 数据结构 / 算法 / SFML 小项目；
- 大一结束~大二：开始把数学课内容写成代码（`Rational`、`Polynomial`、矩阵分解），
  随后在上面手写 MLP 与 CNN（国庆假期基本天天在写）；
- 2026-10：把热点算子移植到 CUDA，并用消融实验做瓶颈分析；
- 2026-10-27~28：手写 GPU 卷积 kernel，修完索引/共享内存越界后与 CPU 版对拍通过；
  加入自适应调度与训练基准。

诚实的教训：**前期没有用版本控制**，10-07 才建立 git 仓库，之前的演进只能靠记忆补记录；
本仓库的 `docs/worklog/`、`BASELINE.md`、标签 `v1.0-baseline` / `v1.1` 都是后来补的"证据链"。
现在的规则是：每次性能改动必须带前后数据，失败实验也要记。

## 路线图

- **v2：设备常驻**（进行中，见 [docs/V2_PLAN.md](docs/V2_PLAN.md)）：把 CNN 的前向/反传整条链
  搬到显存，目标 epoch ≤ 4 s（当前 10.2 s）；里程碑 M0~M4，每个都有独立验收与止损点。
- 计划中的输出（学习记录向）：
  1. 《从零实现一个神经网络：为什么值得写一遍》
  2. 《手写 CUDA GEMM：从 4×4 分块到 LDS 瓶颈》
  3. 《为什么我的 GPU 卷积只快了 1.3 倍：算术强度与 Amdahl》

## 文档索引

| 文档 | 内容 |
|---|---|
| [BASELINE.md](BASELINE.md) | v1.0 GPU GEMM 基线（4×4 分块）与瓶颈分析 |
| [CHANGELOG.md](CHANGELOG.md) | 版本更新记录与版本约定 |
| [docs/V2_PLAN.md](docs/V2_PLAN.md) | v2 设备常驻计划、验收指标、里程碑 |
| [docs/PITFALLS.md](docs/PITFALLS.md) | 踩坑记录（构建、索引、LDS、共享内存越界、数据集身份等） |
| [docs/DEMO_CPU.md](docs/DEMO_CPU.md) | CPU 演示版（无 CUDA/SFML/数据集的快速评估） |
| [docs/CODE_TOUR.md](docs/CODE_TOUR.md) | 代码导览（给读代码的同学：阅读顺序与设计决策） |
| [docs/worklog/](docs/worklog/) | 开发日志（目标 / 假设 / 数据 / 结论） |
| [docs/maintenance/](docs/maintenance/) | 维护日志（构建、测量工具等基础设施修复，与功能推进分开记） |

## 关于 AI 辅助

开发过程中使用了 AI 辅助：代码审查、问题定位、部分样板代码与文档润色。
架构选择、性能实验、调试与数据记录由本人完成。这个声明写在这里，是因为
**"哪些是我判断的"比"代码是谁敲的"更重要**。

## License

个人学习项目，仅用于学习与展示。

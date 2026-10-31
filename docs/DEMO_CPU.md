# CPU 演示版（给同学 / 老师快速评估用）

> 目的：**不需要 NVIDIA 显卡、不需要 CUDA、不需要 SFML、不需要下载数据集**，
> 只用 Visual Studio 打开就能编译运行，用约 1 秒看到这个项目"从零实现"了什么、大概什么水平。

## 一、怎么运行

1. 解压压缩包到任意目录（路径最好不含中文，避免个别环境问题）；
2. 用 **VS 2022 / 2026** 打开 `Demo_CPU.vcxproj`（双击即可；若提示"重定目标"，点确定）；
3. 顶部选择 **Release | x64**，按 **Ctrl + F5**（开始执行不调试，控制台不会一闪而过）。

不需要安装任何第三方库。工程默认使用本机安装的最新 MSVC 工具集（VS2022 用 v143，VS2026 用 v145）。

## 二、预期输出（示例）

> 数字会随机器不同而变化，下面是在 i7-14650HX（24 硬件线程）上的结果。

```
==============================================================
 高等代数 —— 从零实现的 C++ 数学库 / 神经网络（CPU 演示版）
 无需 CUDA / SFML / 数据集；仅需 Visual Studio 与 C++20
==============================================================
[环境] 硬件线程数 = 24

===== 1. 数学库 =====
[Rational] 1/2 + 1/3 + 1/6 = 1/1，精确等于 1 ? 是（double 版本 = 1）
[Matrix] CPU 矩阵乘法基准（单次）：
  256^3 : 0.25 ms（3 次最优）, 134.7 GFLOP/s
  512^3 : 2.26 ms（3 次最优）, 118.7 GFLOP/s
  1024^3 : 33.6 ms（3 次最优）, 63.9 GFLOP/s

===== 2. MLP（从零实现，合成 XOR 数据）=====
  epoch 1  训练耗时 4 ms  准确率 0.93875
  ...
  epoch 15 训练耗时 2 ms  准确率 0.98+

===== 3. CNN（从零实现，合成 8x8 条形图分类）=====
  epoch 1  训练耗时 35 ms  准确率 0.5
  ...
  epoch 12 训练耗时 38 ms  准确率 0.978333

===== 总计 =====
演示总耗时 ~0.7 s
```

## 三、这个演示展示了什么

| 部分 | 内容 | 对应源码 |
|---|---|---|
| 数学库 | `Rational` 精确有理数（1/2+1/3+1/6 精确等于 1，而不是 0.999…） | `Rational.h` |
| 数学库 | `Matrix_Double`：行主序连续存储 + OpenMP + AVX2 + 寄存器分块的 CPU GEMM，千兆次规模下几十 GFLOP/s | `Matrix.h` / `Matrix_Double.cpp` |
| MLP | 从零实现的多层感知机（前向、反向传播、ReLU、Softmax 交叉熵），在合成 XOR 数据上训到 ~99% | `NeuralNetwork_MLP.cpp` |
| CNN | 从零实现的多通道卷积网络（im2col + GEMM 前向、完整反向、maxpool 索引缓存），在含 15% 噪声的 8×8 条形图分类上训到 ~98% | `NeuralNetwork_MULCNN.cpp` / `Batch_Matrix.cpp` |
| 训练 | 网络容器、损失函数、SGD 更新全部手写 | `Complex_NeuralNetwork.cpp` / `Loss.cpp` |

**"从零"的含义**：没有 Eigen / BLAS / cuBLAS / cuDNN / PyTorch；矩阵、卷积、池化、反向传播都是自己写的。

## 四、和完整工程的关系

完整工程（仓库根目录的 `高等代数.vcxproj`）在 CPU 版基础上多了：

- CUDA 手写 GEMM（4096³ 约 **5.48 TFLOPS**，见 `BASELINE.md`）；
- CUDA 手写卷积 kernel（比 CPU 前向快 **7~11 倍**）；
- MNIST/Fashion-MNIST 训练（约 98% 准确率）与自定义手写数字画板（SFML）；
- 自适应 CPU/GPU 调度与测量工具 `Bench`。

完整工程需要：**NVIDIA 显卡 + CUDA ≥ 12.8 + SFML 3.1.0 + 数据集**，门槛较高，所以另做了这个 CPU 演示版。

## 五、限制（如实说明）

- 演示里是**小规模合成任务**，只验证"实现正确、能训练"，不代表 MNIST 最终精度；
- CPU 版不含 MNIST 数据集（52 MB，不入库）；
- 性能数字是 CPU 单机结果，且随机器差异较大；
- 完整 GPU 版中部分模块仍在重构（见 `docs/V2_PLAN.md`），演示版只包含 CPU 部分。

## 六、常见问题

| 问题 | 解决 |
|---|---|
| VS 提示"重定目标项目" | 选"确定"即可（工程会自动用本机工具集） |
| 找不到 v143 / v145 工具集 | VS Installer 勾选"使用 C++ 的桌面开发"工作负载 |
| 控制台中文乱码 | 程序已设置 `SetConsoleOutputCP(CP_UTF8)`；请在 VS 控制台或 Windows Terminal 运行 |
| 运行太快，窗口一闪而过 | 用 **Ctrl + F5**（不要直接双击 exe） |
| 想跑 Debug | 可以，但性能数字没有参考价值（未开优化） |
| CPU 太老不支持 AVX2 | 在 `Demo_CPU.vcxproj` 的 Release 配置里把 `EnableEnhancedInstructionSet` 改为 `NotSet` |

## 七、打包发送建议

只需发送以下文件（不要带 `x64/`、`.vs/`、`*.idx*-ubyte`，否则压缩包会很大）：

```
Demo_CPU.vcxproj
main_cpu_demo.cpp
Loss.h / Loss.cpp
CpuFallback.cpp
Rational.h
Complex.h / Complex.cpp
Polynomial.h / Polynomial.cpp
Matrix.h
Matrix_Double.cpp / Matrix_Complex.cpp / Matrix_Lambda.cpp
Batch_Matrix.h / Batch_Matrix.cpp
NeuralNetwork.h
NeuralNetwork_MLP.cpp / NeuralNetwork_MULCNN.cpp / NeuralNetwork_CNN.cpp
Complex_NeuralNetwork.cpp
README.md（可选）
docs/DEMO_CPU.md（本文件）
```

源码总量约 300 KB，压缩后约 100 KB。

# 高等代数 — 自研 C++ 数学库 / 神经网络 / CUDA 加速

从零实现的 C++ 数值计算库（无第三方依赖），在其上手工搭建 MLP 与多通道 CNN
完成 MNIST 手写数字识别，并把核心热点算子移植到 CUDA。

## 目录

- [项目结构](#项目结构)
- [项目亮点](#项目亮点)
- [性能数据](#性能数据)
- [构建](#构建)
- [踩坑记录](#踩坑记录)

## 项目结构

| 模块 | 文件 | 说明 |
|---|---|---|
| 代数基础 | `Rational` / `Complex` | 精确有理数；复数实虚部均为有理数 |
| 多项式 | `Polynomial` | Karatsuba 分治乘法 + FFT 乘法 |
| 矩阵族 | `Matrix_Complex` / `Matrix_Lambda` / `Matrix_Double` | 初等变换、高斯消元、对角化、Smith 标准型、正交分解 |
| 符号计算 | `MathFunction` | 表达式树 + 符号微分 |
| 神经网络 | `NeuralNetwork_MLP` / `NeuralNetwork_MULCNN` | 见下 |
| CUDA | `Matrix_Double_Mutiple_GPU.cu` | 手写 SGEMM kernel |

## 项目亮点

### 1. 数学库

- `Rational`：精确有理数（分子/分母自动约分），从根源消除浮点舍入误差
- `Polynomial`：Karatsuba 分治乘法（阈值实测 128/256 最优）+ FFT 乘法，复杂度 O(n log n)
- `Matrix_Double`：行主序一维存储，OpenMP + AVX2 + 寄存器分块，1024³ 达 37 GFLOPS
- 正交分解：Householder 三对角化 + 带 Wilkinson 位移的隐式 QR 迭代（特征值分解）
- `MathFunction`：表达式树符号微分，含常数折叠与依赖剪枝

### 2. 神经网络（从零实现）

- `NeuralNetwork_MLP`：多层感知机，输出层用 `None` 激活 + 外部 softmax 交叉熵（避免自行推导 softmax 雅可比）
- `NeuralNetwork_MULCNN`：多通道卷积网络，前向走 **im2col + GEMM**，反向完整实现卷积转置、maxpool argmax 索引缓存
- `Complex_NeuralNetwork`：线性拓扑的网络容器
- MNIST argmax 准确率 **98%+**

### 3. 性能工程

- 训练一轮耗时：**45 s → <10 s**
- 独立重新发明 im2col + GEMM，推导出 `bytes/FLOP = 8 / C_out` 判据，
  用于判断小输出通道时是否该退回朴素卷积（`C_out = 1` 时 im2col 的读放大达 8×）
- 每次性能改动都能归因到具名瓶颈（内存墙 / 并行墙 / 算力墙）

### 4. CUDA

- 手写 SGEMM：共享内存 tiling + 4×4 寄存器分块 + 合并访存 + 设备缓冲常驻
- 建立了 **ABLATION_MODE 消融实验**方法：`0` 完整 / `1` 只加载 / `2` 只计算，
  把加载阶段与计算阶段分离计时，避免"猜瓶颈"
- 用"工作集缩到 L2 后测速不变"的实验**证伪了 DRAM 带宽假设**，
  定位真正瓶颈为 **LDS 指令条数**

## 性能数据

### GPU GEMM（4096³，RTX 5060 Laptop，double→float 边界转换）

| 阶段 | 耗时 |
|---|---|
| MODE=1 加载 | 7.82 ms |
| **MODE=2 计算** | **21.87 ms** |
| **MODE=0 完整** | **25.09 ms** |

编译期资源：`regs = 48`、`local = 0 B`（无 spill）、`blocks/SM = 4`、`threads/SM = 1024`

**瓶颈分析**：计算阶段受共享内存吞吐限制 —— 每轮 8 条 LDS 配 16 次 FMA（1:2），
而硬件 LDS:FMA 吞吐为 1:4 ⇒ **结构上限 = FP32 峰值的 50%**（约 17.2 ms），实测已达该上限的 79%。

### CPU GEMM（1024³ × 100 次，i7-14650HX，OpenMP + AVX2）

| 线程数 | 耗时 | 算力 |
|---|---|---|
| 1 | 34389 ms | 6.2 GFLOPS |
| 8 | 8386 ms | 25.6 GFLOPS |
| 24 | 5762 ms | 37.3 GFLOPS |

## 构建

- **Visual Studio 2022+**（C++20）
- **CUDA Toolkit ≥ 12.8** —— 目标架构 `sm_120`（RTX 50 系），低于 12.8 会报 `no kernel image`
- 必须使用 **Release | x64** 配置（Debug 未启用 AVX2/OpenMP，性能无参考价值）

> ⚠️ **修改 `.cu` 文件后必须执行「生成 → 重新生成解决方案」（Rebuild）。**
> VS 的增量编译检测不到 `.cu` 变化，只点「生成」会链接到旧 kernel。

## 踩坑记录

> 这部分是本文档最有价值的部分 —— 每个坑都花过真实时间。

### 1. 「构建成功」≠「改动生效」

VS 增量编译不检测 `.cu` 变化。更隐蔽的是：**语法合法的旧实现同样编译通过**，
所以"构建成功"完全没有信息量。验证改动必须**读盘核对文件内容**。

### 2. 行列步长常量互换（潜伏 bug）

```cpp
// 错误写法
int RowOut = (blockIdx.y * kBlockSize + ty) * kColsPerThr;  // 行用了列步长
int ColOut = (blockIdx.x * kBlockSize + tx) * kRowsPerThr;  // 列用了行步长
```

4×4 分块时 `kColsPerThr == kRowsPerThr == 4`，错误被**完全掩盖**；
改成 4×8 后立即爆发 —— **矩阵下半部分（行 ≥ N/2）全部为 0**，
原因是行号被放大 2 倍，越界守卫提前一倍触发。

**教训**：行列用同一个常量"碰巧正确"，是最危险的代码。

### 3. 瓶颈不是 DRAM，是 LDS 指令条数

用"工作集缩小到 L2 后测速不变"证伪 DRAM 带宽假设。
真正原因：内层每轮 8 条 LDS 配 16 次 FMA（1:2），而硬件 LDS:FMA 吞吐为 1:4。
**减少字节数（如 double→float）不会改变指令条数，因此完全无效。**

### 4. 4×8 分块为什么反而更慢

`CTXIT = kColsPerThr * tx * kInnerPerT` 使同 warp 相邻线程的地址步长变为
`kColsPerThr × kInnerPerT`：

| 分块 | 步长 | `mod 32` | `gcd(·,32)` | 冲突 | 每单位计算 LDS 周期 |
|---|---|---|---|---|---|
| 4×4 | 188 | 28 | 4 | 2 路 | 0.75 |
| 4×8 | 504 | 24 | **8** | **4 路** | **1.125** |

预测 33.0 ms，实测 35.18 ms（误差 6%，由 occupancy 从 32→24 warp 解释）。

### 5. 数据集身份

排查"训练不收敛"数天后发现：文件名是 MNIST，内容其实是 **Fashion-MNIST**
（官方刻意做成 MNIST 的平替，格式与魔数完全一致）。

**教训**：模型不 work 时，**先看数据**，而不是先改网络结构。

### 6. `Matrix_Double(int size)` 构造的是方阵

`Matrix_Double(9216)` 创建的是 **9216×9216 方阵**（约 679 MB），不是长度 9216 的一维数组。

## License

个人学习项目，仅用于学习与展示。

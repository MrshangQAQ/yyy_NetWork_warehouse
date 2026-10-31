# 数据集：MNIST（原始手写数字，IDX 格式）

本目录包含 4 个文件，沿用 MNIST 的 IDX 文件命名约定（数据经渲染确认为**原始 MNIST 手写数字**，
例如前三个训练样本的标签分别为 5、0、4）：

| 文件 | 内容 |
|---|---|
| `train-images.idx3-ubyte` | 60000 张 28×28 灰度图（每像素 0~255） |
| `train-labels.idx1-ubyte` | 60000 个训练标签（0~9） |
| `t10k-images.idx3-ubyte`  | 10000 张测试图 |
| `t10k-labels.idx1-ubyte`  | 10000 个测试标签 |

## 来源与使用条款

- **来源**：MNIST Database of Handwritten Digits，由 Yann LeCun、Corinna Cortes、Christopher Burges 整理，
  官方页面：<http://yann.lecun.com/exdb/mnist/>
- **原始出处**：数据取自 NIST（美国国家标准与技术研究院）的手写字符数据库
  （NIST Special Database 19），由上述作者整理/裁剪成 MNIST。
- **许可/条款**：官方页面**未声明 OSI 式许可**。该数据集长期用于学术研究；
  本仓库仅在**学习/研究**目的下附带，版权归原作者 / NIST 等原始来源方所有。
  如需商用或再分发，请以官方页面说明为准；若权利方要求，可移除 `data/` 下的数据文件。
- **引用建议**：
  Y. LeCun, L. Bottou, Y. Bengio, P. Haffner.
  *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 1998.

## 格式说明

- `idx3`：大端序头 16 字节（magic=2051, 数量, 行数, 列数），随后逐张 28×28 像素；
- `idx1`：大端序头 8 字节（magic=2049, 数量），随后逐字节标签；
- 像素值 0~255，程序读取时归一化到 [0,1]。

## 补充说明

- 仓库忽略目录 `Train_data/Fasion/` 中曾放过一份 **Fashion-MNIST** 副本（从未启用、不入库）；
  当前 `data/` 使用的是**原始 MNIST**，两者文件名/格式相同，容易混淆，详见 `docs/PITFALLS.md` 第 5 条。
- 程序会自动在 `data/`、仓库根目录、`Train_data/Fasion/` 等位置查找
  `train-images.idx3-ubyte`，无需改代码。

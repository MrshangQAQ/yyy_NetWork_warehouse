# 数据集：Fashion-MNIST（以 MNIST IDX 格式存放）

本目录包含 4 个文件，沿用 MNIST 的 IDX 文件命名约定：

| 文件 | 内容 |
|---|---|
| `train-images.idx3-ubyte` | 60000 张 28×28 灰度图（每像素 0~255） |
| `train-labels.idx1-ubyte` | 60000 个训练标签（0~9） |
| `t10k-images.idx3-ubyte`  | 10000 张测试图 |
| `t10k-labels.idx1-ubyte`  | 10000 个测试标签 |

## 来源与许可

- **数据来源**：这批数据实际是 **Fashion-MNIST**（文件名与格式沿用 MNIST 约定），
  由 Zalando SE 发布：<https://github.com/zalandoresearch/fashion-mnist>
- **许可**：Fashion-MNIST 以 **MIT License** 发布（许可证全文见本目录 `LICENSE-Fashion-MNIST`）；
  转载/再分发时请保留该许可与署名。
- **论文**：Han Xiao, Kashif Rasul, Roland Vollgraf,
  *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, 2017.
- 原始 **MNIST**（手写数字）由 Yann LeCun、Corinna Cortes、Christopher Burges 发布：
  <http://yann.lecun.com/exdb/mnist/> ；如需替换为原始 MNIST，按其页面说明使用。

## 在代码中的使用

程序会自动在 `data/`、仓库根目录、`Train_data/Fasion/` 等位置查找
`train-images.idx3-ubyte`，无需改代码。详见 `README.md` 的『数据集与许可』一节。
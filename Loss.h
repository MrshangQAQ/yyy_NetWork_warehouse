#pragma once
#include "Matrix.h"

/// <summary>计算单个样本交叉熵对 logits 的梯度：∂L/∂z = softmax(z) − y（配合外部学习率使用）。</summary>
Matrix_Double CrossEntropyGradient(const Matrix_Double& z, const Matrix_Double& y);

/// <summary>逐样本交叉熵梯度（三阶张量版本）：对每个样本矩阵分别计算 softmax − y。</summary>
std::vector<Matrix_Double> CrossEntropyGradient(const std::vector<Matrix_Double>& z,
                                                const std::vector<Matrix_Double>& y);

/// <summary>交叉熵损失：-Σ y·log(softmax(z))；abs(y) &lt; approximation 的项跳过（避免 log(0)）。</summary>
double CrossEntropy(const Matrix_Double& z, const Matrix_Double& y, double approximation = 0.01);

// CPU 演示工程专用：为 GPU 接口提供 CPU 版本的空实现 / 回退实现。
// 只有在定义 CPU_ONLY 的工程（Demo_CPU.vcxproj）里才参与编译，
// 完整工程（高等代数.vcxproj）由对应的 .cu 提供真正的 GPU 实现。
#ifdef CPU_ONLY

#include "Matrix.h"
#include <utility>

/// <summary>CPU 回退：矩阵乘法自动选择接口在 CPU 版工程里直接走 CPU。</summary>
void Matrix_Double_Mutiple_Auto(const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out) {
	Matrix_Double tmp = mat1 * mat2;
	out = std::move(tmp);
}

/// <summary>CPU 版工程没有 GPU 阈值可调，保留空实现以满足接口。</summary>
void SetMatMulGpuThreshold(size_t) {}

/// <summary>CPU 版工程没有传输路径可切换，保留空实现以满足接口。</summary>
void SetMatMulTransferMode(int) {}

#endif // CPU_ONLY

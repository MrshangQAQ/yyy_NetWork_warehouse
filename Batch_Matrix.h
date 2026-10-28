#pragma once
#include"Matrix.h"
#include<cmath>

/// GPU 瘦 N 卷积 kernel 支持的输出通道数上限。
/// 与 imcol2_GPU.cu 中 GemmAccum_ThinN_Kernel 的共享内存数组尺寸保持一致，改一处必须同步改另一处。
constexpr int kGemmThinNMax = 16;

class Batch_Matrix
{
public:
	//构造函数
	Batch_Matrix(std::vector<Matrix_Double> data_,int rownum,int colnum);//当data_不足时，会自动补0矩阵
	Batch_Matrix(int rownum, int colnum,int interRownum,int interColnum);//自动生成空的Batch矩阵
	explicit Batch_Matrix(int size=0, int interSize=1) ;//自动生成空的Batch矩阵
	//简单的数据操作
	int GetRowNum()const { return rowNums_; };
	int GetColNum()const { return colNums_; };
	const std::vector<Matrix_Double>& GetInternalData() const { return data; };
private:
	int colNums_;
	int rowNums_;
	std::vector<Matrix_Double> data;
public:
	friend Batch_Matrix operator+(const Batch_Matrix& bat1, const Batch_Matrix& bat2);
	friend Batch_Matrix& operator+=(Batch_Matrix& bat1, const Batch_Matrix& bat2);
	friend Batch_Matrix operator-(const Batch_Matrix& bat1, const Batch_Matrix& bat2);
	friend Batch_Matrix& operator-=(Batch_Matrix& bat1, const Batch_Matrix& bat2);
public:
	void InterRotato_180();//将类内的每个元素都旋转180度
	void Random(double ave, double std);//初始随机化
	/// <summary>对批内每个块做原地最大池化，返回每块的反池化上下文，顺序与内部块一致。</summary>
	/// <param name="pollsize">池化窗口边长 k，须为正整数</param>
	/// <returns>上下文数组，长度 = 批内块数；第 i 个元素对应内部 data[i]。</returns>
	/// <exception cref="std::runtime_error">pollsize&lt;=0 时抛出</exception>
	std::vector<AntiPoll_Data> MaxPoll(int pollsize);
public:
	friend Batch_Matrix GetConvolutionMatrixMul(const Batch_Matrix& bat, const Batch_Matrix& ker, int padding, int stride);
	friend Batch_Matrix GetBlockTranspose(const Batch_Matrix& bat);
	friend Batch_Matrix MapMatrix(const Batch_Matrix& bat, double (*f)(double));
	friend Batch_Matrix Hadamard(const Batch_Matrix& mat1, const Batch_Matrix& mat2);
	friend Batch_Matrix operator*(double db, const Batch_Matrix& bat);
	friend Batch_Matrix& operator*=(double db, Batch_Matrix& bat);
	friend void GetConvolutionMatrixMul_GPU(const Batch_Matrix& bat, const Batch_Matrix& ker, Batch_Matrix& out, int padding, int stride);
};
Batch_Matrix operator+(const Batch_Matrix& bat1, const Batch_Matrix& bat2);
Batch_Matrix& operator+=(Batch_Matrix& bat1, const Batch_Matrix& bat2);
Batch_Matrix operator-(const Batch_Matrix& bat1, const Batch_Matrix& bat2);
Batch_Matrix& operator-=(Batch_Matrix& bat1, const Batch_Matrix& bat2);
Batch_Matrix operator*(double db, const Batch_Matrix& bat);
Batch_Matrix& operator*=(double db, Batch_Matrix& bat);
/// 类似于矩阵乘法的卷积
Batch_Matrix GetConvolutionMatrixMul(const Batch_Matrix& bat,const Batch_Matrix& ker,int padding=0, int stride=1);

/// GPU 版多管道卷积（实现在 imcol2_GPU.cu）：与 GetConvolutionMatrixMul 数值等价，
/// 内部把 im2col 展开与矩阵乘都放在显存里完成（FP32 计算、double 存储）。
/// <param name="bat">输入图网格：行 × 输入通道，每块一张特征图</param>
/// <param name="ker">核网格：输入通道 × 输出通道，每块 K×K</param>
/// <param name="padding">四周补零的行列数</param>
/// <param name="stride">滑窗步长</param>
/// <returns>输出网格：bat 的行数 × 输出通道</returns>
/// <exception cref="std::runtime_error">通道数不匹配、核不是方阵，或 C_out 超过 kGemmThinNMax 时抛出</exception>
void GetConvolutionMatrixMul_GPU(const Batch_Matrix& bat, const Batch_Matrix& ker,Batch_Matrix& out, int padding=0, int stride=1);

/// 块转置：把 (Co×Ci) 的核网格翻成 (Ci×Co)（反向传播要用）
Batch_Matrix GetBlockTranspose(const Batch_Matrix& bat);

///映射函数，Batch中的每一个元素都进行一次Map
Batch_Matrix MapMatrix(const Batch_Matrix& bat,double (*f)(double));

//将最终得到的Batch_Matrix的每一行压缩成一个矩阵，得到第i个新矩阵的第j列对应了原来的第i行中的第j个矩阵，然后将之返回
std::vector<Matrix_Double> FuseKernelMaps(const Batch_Matrix& kernelOut);

///将反向传播的三阶张量，拼成Batcj_Matrix，按内部矩阵的每一列展开为一个新的矩阵，内部第i个矩阵的第j列，对应了得到的Batch矩阵的第i行的第j个矩阵,newColNum表示了新矩阵的列数
///关于为什么考虑newColNum，因为列数对应了多管道卷积最后得到的特征数，而行数则表示了参数的大小，而列数可以直接算出来，行数不行
Batch_Matrix SplitKernelGrad(const std::vector<Matrix_Double>& grad, int newInterColNum);

//Batch_Matrix的逐元素乘法
Batch_Matrix Hadamard(const Batch_Matrix& mat1, const Batch_Matrix& mat2);

//为内部的每个矩阵周边添加padding行（与列）0元素
Batch_Matrix GetPaddingMatrix(const Batch_Matrix& bat, int padding);

//为内部的每个矩阵矩阵的每个元素之间插入stride-1个元素
Batch_Matrix GetDilateMatrix(const Batch_Matrix& bat, int stride);

/// <summary>批量反池化：按每块上下文把梯度网格还原成池化前的块内尺寸，返回新 Batch_Matrix。网格尺寸与输入 dA 相同（池化不改变样本/通道网格）。</summary>
/// <param name="dA">池化后的梯度批矩阵</param>
/// <param name="pollsize">池化窗口边长 k（须与 MaxPoll 时一致）</param>
/// <param name="ctxs">MaxPoll 返回的上下文数组，长度须等于批内块数</param>
/// <returns>块内尺寸还原为池化前尺寸的新 Batch_Matrix，样本/通道网格沿用 dA</returns>
/// <exception cref="std::runtime_error">pollsize&lt;=0，或上下文数量与批不匹配时抛出</exception>
Batch_Matrix GetAntiMaxPoll(const Batch_Matrix& dA, int pollsize, const std::vector<AntiPoll_Data>& ctxs);

/// <summary>自适应卷积：按乘加数自动选择 CPU/GPU，结果写入 out；GPU 路径复用 out 的块，避免重复分配。</summary>
/// <param name="bat">输入图网格：行 × 输入通道</param>
/// <param name="ker">核网格：输入通道 × 输出通道</param>
/// <param name="out">输出网格，尺寸不符时会被调整为 R × C_out</param>
/// <param name="padding">四周补零的行列数</param>
/// <param name="stride">滑窗步长</param>
void GetConvolutionMatrixMul_Auto(const Batch_Matrix& bat, const Batch_Matrix& ker, Batch_Matrix& out, int padding = 0, int stride = 1);

/// <summary>设置卷积的 GPU 启用阈值（macs = R×L×(C_in·K²)×C_out）；传 SIZE_MAX 可全部走 CPU。</summary>
void SetConvGpuThreshold(size_t macs);
/// <summary>估算 GPU 卷积会开出的总线程数（用于判断这一个形状值不值得上 GPU）。</summary>
size_t GetConvGpuThreadCount(int R, int C_out, int outRows, int outCols);
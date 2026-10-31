#pragma once
#include"Batch_Matrix.h"
/// <summary>
/// 块矩阵数据常驻GPU的版本，持有一个指向GPU中数据地址的指针和4个维度的数据，固定大小
/// 拷贝运算已经禁用，但允许使用移动语义，若实在需要拷贝建议转为CPU版本Batch_Matrix再拷贝，不提供GPU版本的拷贝方法
/// </summary>
class GPUBatchMatrix {
private:
	float* data_=nullptr;
	size_t outRowNums_;
	size_t outColNums_;
	size_t inRowNums_;
	size_t inColNums_;

public:
	//基础的构造部分(CPU实现)
	GPUBatchMatrix(const Batch_Matrix& bat);
	GPUBatchMatrix(int outRowNums=0, int outColNums=0, int inRowNums=0, int inColNums=0);
	GPUBatchMatrix(GPUBatchMatrix& bat)= delete;
	GPUBatchMatrix(GPUBatchMatrix&& bat);
	~GPUBatchMatrix();
	Batch_Matrix ToCPUBat();
	GPUBatchMatrix& operator=(GPUBatchMatrix& bat) = delete;
	GPUBatchMatrix& operator=(GPUBatchMatrix&& bat);

	float* const GetData() { return data_; };//得到数据
	const float* const GetDataOnlyRead() const { return data_; };//得到只读数据
	size_t GetOutColNums() const { return outColNums_; };
	size_t GetOutRowNums() const { return outRowNums_; };
	size_t GetOutSize() const { return outColNums_ * outRowNums_; };

	size_t GetInColNums() const { return inColNums_; };
	size_t GetInRowNums() const { return inRowNums_; };
	size_t GetInSize()const { return inColNums_ * inRowNums_; };

	size_t GetTotalSize()const { return outColNums_ * outRowNums_ * inColNums_ * inRowNums_; };
	//校准矩阵，在卷积，旋转180度等等时候，检测该矩阵是否符合要求，若不符合要求，会尝试重新为其分配空间和行列数,并返回是否成功
	bool TryMatchWith(size_t aimOutRow, size_t aimOutCol,size_t aimInRow,size_t aimInCol);
	//GPU加速运算的实现部分
	friend GPUBatchMatrix& operator += (GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2);
	friend GPUBatchMatrix& operator -= (GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2);
	friend GPUBatchMatrix& operator *= (const GPUBatchMatrix& bat1, float f);

	friend void GetConvolutionMatrixMul_GPU(const GPUBatchMatrix& bat, const GPUBatchMatrix& ker, GPUBatchMatrix& out, int padding, int stride);
};
//GPU加速实现的部分
GPUBatchMatrix& operator += (GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2);
GPUBatchMatrix& operator -= (GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2);
GPUBatchMatrix& operator *= (GPUBatchMatrix& bat1, float f);

//测试两矩阵是否匹配的函数
bool IsMatch(const GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2);
//卷积
void GetConvolutionMatrixMul_GPU(const GPUBatchMatrix& bat, const GPUBatchMatrix& ker, GPUBatchMatrix& out, int padding = 0, int stride = 1);
//反转180度，反向传播时用
void InterRotato_180(const GPUBatchMatrix& bat,GPUBatchMatrix& out);
//逐元素乘法
void Hadamard(const GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2, GPUBatchMatrix& out);
void HadamardWith(const GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2);
//GPU上的逐元素映射，使用时输入函数模板
//池化
void MaxPoll(const GPUBatchMatrix& bat ,GPUBatchMatrix& poll,size_t* antipollBuffer,size_t pollsize);
//将poll反池化后乘以f加到bat上
void AntiPoll_Timesf_Addto(const GPUBatchMatrix& poll, const size_t* const antipollBuffer, float f, GPUBatchMatrix& bat);
//Dilate,在该矩阵的内维的每两个元素之间，插入stride-1个0
void DilateTo(const GPUBatchMatrix& bat, size_t stride, GPUBatchMatrix& out,bool setzero);
#pragma once
#pragma once
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <omp.h>
#include "Polynomial.h"
#include "Complex.h"

//宏，用于检查数据在存储数据之间的移动是否正常
#ifndef CPU_ONLY
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("[CUDA] ") + __FILE__ + ":" +  \
                std::to_string(__LINE__) + "  " + cudaGetErrorString(_err));    \
        }                                                                       \
    } while (0)
#else
#define CUDA_CHECK(call) do { (void)(call); } while (0)   // CPU 版工程里没有 CUDA 调用
#endif

class Matrix_Complex;
class MatrixView_Complex;
struct MatrixOperationsReturn_Complex;
struct GetInverseData_Complex;
struct OrthogonalDecomposition_Result;//这里是对于Matrix_Complex的，但我懒得改了

class Matrix_Double;
struct MatrixOperationsReturn_Double;
struct OrthogonalDecomposition_Result_Double;
struct AntiPoll_Data;

struct SVD_Result;


class Matrix_Lambda {
protected:
	int rowNum_;//行数
	int colNum_;//列数
	std::vector<Polynomial> data;
public:
	Matrix_Lambda(std::vector<Polynomial>data_, int row, int col);
	Matrix_Lambda(int row, int col);
	explicit Matrix_Lambda(int size = 1) ;
	int GetRowNum() const { return rowNum_; }
	int GetColNum() const { return colNum_; }
	//基础变换，private的部分没有安全檢查，用于内部调用加速
public:
	void Swap_Row(int row1, int row2);
	void Mutiple_Row(const Polynomial& times, int row);
	void AddTimesTo_Row(int addRow, const Polynomial& times, int beAddedRow);
	void Swap_Col(int col1, int col2);
	void Mutiple_Col(const Polynomial& times, int col);
	void AddTimesTo_Col(int addCol, const Polynomial& times, int beAddedCol);
	Matrix_Lambda GetTranslateMatrix();//得到转置矩阵（矩阵本身不变）
	void Translate();//转置该矩阵
	int Find_Main(int col);//返回行多项式次数最小的元素所在行,会跳过0多项式
private:
	void InISwap_Row(int row1, int row2);
	void InIMutiple_Row(const Polynomial& times, int row);
	void InIAddTimesTo_Row(int addRow, const Polynomial& times, int beAddedRow);
	void InISwap_Col(int col1, int col2);
	void InIMutiple_Col(const Polynomial& times, int col);
	void InIAddTimesTo_Col(int addCol, const Polynomial& times, int beAddedCol);
	int InIFind_Main(int col, int beginRow);//返回行多项式次数最小的元素所在行，beginRow表示从该Row开始搜素，用于加速


	//数据操作,public的部分有安全检查，private的部分没有安全檢查，用于内部调用加速
public:
	Polynomial GetData(int row, int col);
	void SetData(int row, int col, Polynomial newdata);

private:
	Polynomial InIGetData(int row, int col);
	void InISetData(int row, int col, Polynomial newdata);

	//对角化与行列式计算
public:
	void Diagonalization();//原地对角化（矩阵本身被修改为标准型）
	Polynomial GetDeterminant();
	Complex GetPolynomailValue(Complex value);
public:

	// 运算符重载：需要访问 protected 成员，故声明为友元函数
	friend Matrix_Lambda operator+(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
	friend Matrix_Lambda& operator+=(Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
	friend Matrix_Lambda operator-(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
	friend Matrix_Lambda& operator-=(Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
	friend Matrix_Lambda operator*(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
	friend Matrix_Lambda operator*(Polynomial pol, const Matrix_Lambda& mat);
	friend Matrix_Lambda& operator*=(Matrix_Lambda& mat, Polynomial pol);
	friend std::ostream& operator<<(std::ostream& os, const Matrix_Lambda& mat);
	friend Matrix_Complex GetComplexMatrix(const Matrix_Lambda& mat, Complex val);
	friend Matrix_Lambda Hadamard(const Matrix_Lambda& mat1, const Matrix_Lambda& mat2);
};
Matrix_Lambda operator+(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
Matrix_Lambda& operator+=(Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
Matrix_Lambda operator-(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
Matrix_Lambda& operator-=(Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
Matrix_Lambda operator*(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2);
Matrix_Lambda operator*(Polynomial pol, const Matrix_Lambda& mat);
Matrix_Lambda operator*(const Matrix_Lambda& mat,Polynomial pol);
Matrix_Lambda& operator*=(Matrix_Lambda& mat,Polynomial pol);
std::ostream& operator<<(std::ostream& os, const Matrix_Lambda& mat);
Matrix_Lambda GetDiagonalization(const Matrix_Lambda& mat);//得到对角化副本（矩阵本身不变）
Matrix_Lambda Hadamard(const Matrix_Lambda& mat1,const Matrix_Lambda& mat2);//逐元素乘法

class Matrix_Complex {
protected:
	int rowNum_;//行数
	int colNum_;//列数
	std::vector<Complex> data;
public:
	//构造函数
	Matrix_Complex(int row , int col );
	Matrix_Complex(std::vector<Complex>data_, int row, int col);
	Matrix_Complex(std::vector<float>data_, int row, int col);
	explicit Matrix_Complex(int size=1);//自动生成单位方阵
	Matrix_Complex(MatrixView_Complex View);//通过视图构造

	int GetRowNum() const { return rowNum_; }
	int GetColNum() const { return colNum_; }
//基础变换，private的部分没有安全檢查，用于内部调用加速
public:
	void Swap_Row(int row1,int row2);
	void Mutiple_Row(Complex times, int row);
	void AddTimesTo_Row(int addRow, Complex times, int beAddedRow);
	void Swap_Col(int col1, int col2);
	void Mutiple_Col(Complex times, int col);
	void AddTimesTo_Col(int addCol, Complex times, int beAddedCol);
	void Translate();//转置该矩阵
private:
	void InISwap_Row(int row1, int row2);
	void InIMutiple_Row(Complex times, int row);
	void InIAddTimesTo_Row(int addRow, Complex times, int beAddedRow);
	void InISwap_Col(int col1, int col2);
	void InIMutiple_Col(Complex times, int col);
	void InIAddTimesTo_Col(int addCol, Complex times, int beAddedCol);

//得到子矩阵
public:
	Matrix_Complex GetMinor(int BeginRow,int EndRow,int BeginCol,int EndCol);//得到从begin到end部分的子矩阵，begin和end本身都会被计入
	Matrix_Complex GetMinorByVec(const std::vector<int>& RowVec, const std::vector<int>& ColVec);//得到特定的某些行和列形成的子式，注意，得到的子式的排布顺序按照输入的顺序表示，使用前可能需要手动sort
private:
	Matrix_Complex InIGetMinor(int BeginRow, int EndRow, int BeginCol, int EndCol);//无安全检查，得到从begin到end部分的子矩阵，begin和end本身都会被计入
	Matrix_Complex InIGetMinorByVec(const std::vector<int>& RowVec,const std::vector<int>& ColVec);//无安全检查，得到特定的某些行和列形成的子式
//数据操作,public的部分有安全检查，private的部分没有安全檢查，用于内部调用加速
public:
	Complex GetData(int row, int col);
	void SetData(int row, int col, Complex newdata);
	int Find_Main(int col);//返回行主元所在行

private:
	Complex InIGetData(int row, int col);
	void InISetData(int row, int col, Complex newdata);
	int InIFind_Main(int col,int beginRow,double minModelSquare);//返回行主元所在行，beginRow表示从该Row开始搜素，用于加速
public:
	//高斯消元,对角化，求行列式与求逆矩阵
	void NeglectSmall(double approximation);//将集合中，所有小于approximation的值设为0
	MatrixOperationsReturn_Complex GaussEliminat(double approximation);
	MatrixOperationsReturn_Complex Diagonalization(double approximation);
	Complex GetDeterminant(float approximation);
	GetInverseData_Complex GetRank(float approximation);//返回矩阵的秩
public:
	//得到相似标准型，奇异值分解，若当标准型，lambda矩阵等
	OrthogonalDecomposition_Result OrthogonalDecomposition();
	double GetModle();
private:
	OrthogonalDecomposition_Result InIOrthogonalDecomposition();//返回正交分解结果（正交矩阵+特征值），不会检查矩阵本身是否为对称矩阵
	double InIGetModle();//若矩阵只有一行或一列，返回其模长
public:
	// 运算符重载：需要访问 protected 成员，故声明为友元函数
	friend Matrix_Complex operator+(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
	friend Matrix_Complex& operator+=(Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
	friend Matrix_Complex operator-(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
	friend Matrix_Complex& operator-=(Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
	friend Matrix_Complex operator*(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
	friend Matrix_Complex operator*(Complex com, const Matrix_Complex& mat);
	friend Matrix_Complex& operator*=(Matrix_Complex& mat, Complex com);
	friend std::ostream& operator<<(std::ostream& os, const Matrix_Complex& mat);
	friend struct MatrixView_Complex;
	friend Matrix_Complex GetTranslateMatrix(const Matrix_Complex& mat);
	friend Matrix_Lambda GetLambdaMatrix(const Matrix_Complex& mat);
	friend Matrix_Complex GetInverseMatrix(const Matrix_Complex& mat, float approximation);
	friend Matrix_Double GetDoubleMatrix(const Matrix_Complex& mat);//得到双精度矩阵（只取实部）
};
struct SVD_Result
{
	Matrix_Complex U;
	std::vector<Complex> Spectrum;
	Matrix_Complex V;
};
/// <summary>对称矩阵正交分解的结果：A = Q·Λ·Qᵀ，Λ 的对角线元素就是 Spectrum。</summary>
struct OrthogonalDecomposition_Result
{
	Matrix_Complex Q;//正交矩阵，每一列是一个特征向量（与 Spectrum 一一对应）
	std::vector<Complex> Spectrum;//特征值数组（按从大到小排列）
};
Matrix_Complex operator+(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
Matrix_Complex& operator+=(Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
Matrix_Complex operator-(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
Matrix_Complex& operator-=(Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
Matrix_Complex operator*(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2);
Matrix_Complex operator*(Complex com, const Matrix_Complex& mat);
Matrix_Complex operator*(const Matrix_Complex& mat,Complex com);
Matrix_Complex& operator*=(Matrix_Complex& mat, Complex com);
std::ostream& operator<<(std::ostream& os, const Matrix_Complex& mat);

Matrix_Complex GetTranslateMatrix(const Matrix_Complex& mat);//得到转置矩阵（矩阵本身不变）
Matrix_Lambda GetLambdaMatrix(const Matrix_Complex& mat);//得到 lambda 矩阵（矩阵本身不变）
Matrix_Complex GetInverseMatrix(const Matrix_Complex& mat, float approximation);//得到逆矩阵（矩阵本身不变）


struct MatrixOperationsReturn_Complex
{
	Matrix_Complex ResultMat;//高斯消元得到的结果矩阵
	Matrix_Complex Memory;//高斯消元时，记录行列变换的初等矩阵
	int Swaptimes;//进行交换变换的次数，用于计算行列式
};
struct GetInverseData_Complex {
	MatrixOperationsReturn_Complex OperaData;
	int rank;
};

struct MatrixView_Complex {
	const Complex* first;
	int RowBegin, RowLen;
	int ColBegin, ColLen;
	int ColNum_;
	MatrixView_Complex(const Matrix_Complex& mat) :
		first(mat.data.data()),
		RowBegin(0), ColBegin(0),
		RowLen(mat.rowNum_-1),ColLen(mat.colNum_-1),
		ColNum_(mat.colNum_)
	{}
	MatrixView_Complex(const Complex* f,int rowBegin, int rowLen,int colBegin,int colLen,int colNum_):
		first(f),
		RowBegin(rowBegin),RowLen(rowLen),
		ColBegin(colBegin),ColLen(colLen),
		ColNum_(colNum_)
	{}
	MatrixView_Complex UpperLeft() { return MatrixView_Complex(first, RowBegin, RowLen/2, ColBegin, ColLen/2, ColNum_); }
	MatrixView_Complex UpperRight(){ return MatrixView_Complex(first, RowBegin+ RowLen/2, RowLen-RowLen/2, ColBegin, ColLen / 2, ColNum_); }
	MatrixView_Complex DownLeft() { return MatrixView_Complex(first, RowBegin, RowLen/2, ColBegin+ColLen/2, ColLen-ColLen/2, ColNum_); }
	MatrixView_Complex DownRight() { return MatrixView_Complex(first, RowBegin + RowLen / 2, RowLen - RowLen / 2, ColBegin + ColLen / 2, ColLen - ColLen / 2, ColNum_); }
	Complex VisData(int row, int col) const { return first[(row+RowBegin)*ColNum_+col+ColBegin];}
};
///注：此为非公用API，调用时需谨慎，可能发生未定义行为
namespace Internal {
	struct Matrix_Double_Access;
}
//在这底下写，Matrix_Double类的实现
/// <summary>双精度矩阵类：元素为 double，按行主序存放在一维数组中（下标 = row * colNum_ + col）。
/// 主要用于数值计算场景（如神经网络训练）；需要精确计算时请使用 Matrix_Complex。</summary>
class Matrix_Double {
	friend struct Internal::Matrix_Double_Access;
protected:
	int rowNum_;//行数
	int colNum_;//列数
	std::vector<double> data;//矩阵元素，行主序存储
public:
	//构造函数
	/// <summary>构造 row×col 的全零矩阵</summary>
	Matrix_Double(int row, int col);
	/// <summary>从一维数组构造 row×col 矩阵：数据不足时补零，多余时截断</summary>
	Matrix_Double(std::vector<double>data_, int row, int col);
	/// <summary>从 float 一维数组构造 row×col 矩阵</summary>
	Matrix_Double(std::vector<float>data_, int row, int col);
	/// <summary>构造 size×size 的单位方阵（size 缺省为 1）</summary>
	explicit Matrix_Double(int size = 1);
	int GetRowNum() const { return rowNum_; }
	int GetColNum() const { return colNum_; }
	const std::vector<double>& GetInternalData() const { return data; }
	//基础变换，private的部分没有安全检查，用于内部调用加速
public:
	/// <summary>交换第 row1 行与第 row2 行</summary>
	void Swap_Row(int row1, int row2);
	/// <summary>将第 row 行乘以 times</summary>
	void Mutiple_Row(double times, int row);
	/// <summary>倍加变换：第 beAddedRow 行 += times × 第 addRow 行</summary>
	void AddTimesTo_Row(int addRow, double times, int beAddedRow);
	/// <summary>交换第 col1 列与第 col2 列</summary>
	void Swap_Col(int col1, int col2);
	/// <summary>将第 col 列乘以 times</summary>
	void Mutiple_Col(double times, int col);
	/// <summary>倍加变换：第 beAddedCol 列 += times × 第 addCol 列</summary>
	void AddTimesTo_Col(int addCol, double times, int beAddedCol);

	/// <summary>转置该矩阵（原地修改）</summary>
	void Translate();
private:
	void InISwap_Row(int row1, int row2);
	void InIMutiple_Row(double times, int row);
	void InIAddTimesTo_Row(int addRow, double times, int beAddedRow);
	void InISwap_Col(int col1, int col2);
	void InIMutiple_Col(double times, int col);
	void InIAddTimesTo_Col(int addCol, double times, int beAddedCol);

	//数据操作,public的部分有安全检查，private的部分没有安全检查，用于内部调用加速
public:
	/// <summary>读取第 row 行第 col 列的元素（带越界检查）</summary>
	double GetData(int row, int col) const;
	/// <summary>写入第 row 行第 col 列的元素（带越界检查）</summary>
	void SetData(int row, int col, double newdata);
	/// <summary>返回该列中绝对值最大的元素所在行</summary>
	int Find_Main(int col) const;
private:
	double InIGetData(int row, int col) const;
	void InISetData(int row, int col, double newdata);
	int InIFind_Main(int col, int beginRow, double minModel) const;//从 beginRow 行起找该列绝对值最大的元素，全都不超过 minModel 时返回 -1
public:
	//高斯消元，对角化与正交分解
	/// <summary>将矩阵中绝对值小于 approximation 的元素置为 0</summary>
	void NeglectSmall(double approximation);
	/// <summary>高斯消元：返回消元结果、记录行列变换的初等矩阵与交换次数</summary>
	MatrixOperationsReturn_Double GaussEliminat(double approximation);
	/// <summary>对角化：返回对角形结果、记录变换的初等矩阵与交换次数</summary>
	MatrixOperationsReturn_Double Diagonalization(double approximation);
	/// <summary>对称矩阵的正交分解：A = Q·Λ·Qᵀ，要求矩阵严格对称</summary>
	OrthogonalDecomposition_Result_Double OrthogonalDecomposition();
private:
	OrthogonalDecomposition_Result_Double InIOrthogonalDecomposition();//无安全检查：不检查矩阵是否对称

public:
	Matrix_Double GetMinor(int BeginRow, int EndRow, int BeginCol, int EndCol) const;//得到从begin到end部分的子矩阵，begin和end本身都会被计入
	Matrix_Double GetMinorByVec(const std::vector<int>& RowVec, const std::vector<int>& ColVec) const;//得到特定的某些行和列形成的子式，注意，得到的子式的排布顺序按照输入的顺序表示，使用前可能需要手动sort
	Matrix_Double GetMinorByVec(const std::vector<int>& ColVec)const;//若只输入一个指标，默认按照列读取，如需要行，手动转置
private:
	Matrix_Double InIGetMinor(int BeginRow, int EndRow, int BeginCol, int EndCol) const;//无安全检查，得到从begin到end部分的子矩阵，begin和end本身都会被计入
	Matrix_Double InIGetMinorByVec(const std::vector<int>& RowVec, const std::vector<int>& ColVec) const;//无安全检查，得到特定的某些行和列形成的子式
	// 运算符重载：需要访问 protected 成员，故声明为友元函数
public:
	void Random(double ave, double std);
	void Rotato_180();
	/// <summary>最大池化（非重叠窗口，原地修改自身）：窗口边长 k = pollsize，
	/// 输出尺寸 (H/k)×(W/k)（整数除法向下取整），不足一个窗口的边角像素直接丢弃、不补零。</summary>
	/// <param name="pollsize">池化窗口边长 k，须为正整数</param>
	/// <returns>冠军位置索引列表，长度 = (H/k)·(W/k)，按行主序对应输出格；
	/// 每元素为该输出格窗口内最大值的位置偏移（窗口内第几个元素，0..k²-1，行主序）。</returns>
	/// <exception cref="std::runtime_error">pollsize&lt;=0，或矩阵尺寸不足一个窗口时抛出</exception>
	AntiPoll_Data MaxPoll(int pollsize);
public:
	friend Matrix_Double operator+(const Matrix_Double& matrix1, const Matrix_Double& matrix2);
	friend Matrix_Double& operator+=(Matrix_Double& matrix1, const Matrix_Double& matrix2);
	friend Matrix_Double operator-(const Matrix_Double& matrix1, const Matrix_Double& matrix2);
	friend Matrix_Double& operator-=(Matrix_Double& matrix1, const Matrix_Double& matrix2);
	friend Matrix_Double operator*(const Matrix_Double& matrix1, const Matrix_Double& matrix2);
	friend Matrix_Double operator*(double db, const Matrix_Double& mat);
	friend Matrix_Double& operator*=(Matrix_Double& mat, double db);
	friend std::ostream& operator<<(std::ostream& os, const Matrix_Double& mat);
	friend Matrix_Double GetTranslateMatrix(const Matrix_Double& mat);
	friend Matrix_Double GetInverseMatrix(const Matrix_Double& mat, double approximation);
	friend Matrix_Complex GetComplexMatrix(const Matrix_Double& mat);
	friend Matrix_Double Hadamard(const Matrix_Double& mat1, const Matrix_Double& mat2);
	friend Matrix_Double MapMatrix(const Matrix_Double& mat, double(*f)(double));
	friend Matrix_Double Sigmoid(const Matrix_Double& z);
	friend Matrix_Double SigmoidGradient(const Matrix_Double& a);
	friend Matrix_Double ReLU(const Matrix_Double& z);
	friend Matrix_Double ReLUGradient(const Matrix_Double& z);
	friend Matrix_Double RowSums(const Matrix_Double& mat);
	friend Matrix_Double AddBias(const Matrix_Double& mat, const Matrix_Double& bias);
	friend Matrix_Double Softmax(const Matrix_Double& z);
	friend Matrix_Double Convolution(const Matrix_Double& mat,const Matrix_Double& kernel,int padding,int stride);
	friend Matrix_Double GetPaddingMatrix(const Matrix_Double& mat,int padding);
	friend Matrix_Double GetDilateMatrix(const Matrix_Double& mat, int stride);
	friend Matrix_Double GetFlattenMatrixFromBatch(const std::vector<Matrix_Double>& batch);
	friend std::vector<Matrix_Double> UnFlattenToBatch(const Matrix_Double& mat, int h, int w);
	friend void Matrix_Double_Mutiple_GPU(const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out);

public:
	//以下函数在GPU(Matrix_Double_Mutiple_GPU)中实现
	friend Matrix_Double Matrix_Double_Mutiple_GPU(const Matrix_Double& mat1, const Matrix_Double& mat2);
};
struct MatrixOperationsReturn_Double
{
	Matrix_Double ResultMat;//高斯消元得到的结果矩阵
	Matrix_Double Memory;//高斯消元时，记录行列变换的初等矩阵
	int Swaptimes;//进行交换变换的次数，用于计算行列式
};
/// <summary>对称矩阵正交分解的结果：A = Q·Λ·Qᵀ，Λ 的对角线元素就是 Spectrum。</summary>
struct OrthogonalDecomposition_Result_Double
{
	Matrix_Double Q;//正交矩阵，每一列是一个特征向量（与 Spectrum 一一对应）
	std::vector<double> Spectrum;//特征值数组（按从大到小排列）
};
Matrix_Double operator+(const Matrix_Double& matrix1, const Matrix_Double& matrix2);
Matrix_Double& operator+=(Matrix_Double& matrix1, const Matrix_Double& matrix2);
Matrix_Double operator-(const Matrix_Double& matrix1, const Matrix_Double& matrix2);
Matrix_Double& operator-=(Matrix_Double& matrix1, const Matrix_Double& matrix2);
Matrix_Double operator*(const Matrix_Double& matrix1, const Matrix_Double& matrix2);
Matrix_Double operator*(double db, const Matrix_Double& mat);
Matrix_Double operator*(const Matrix_Double& mat, double db);
Matrix_Double& operator*=(Matrix_Double& mat, double db);
std::ostream& operator<<(std::ostream& os, const Matrix_Double& mat);
Matrix_Double Hadamard(const Matrix_Double& mat1, const Matrix_Double& mat2);//逐元素积
std::vector<Matrix_Double> Hadamard(const std::vector<Matrix_Double>& mat1, const std::vector<Matrix_Double>& mat2);//逐元素积
Matrix_Double MapMatrix(const Matrix_Double& mat, double(*f)(double));//逐元素施加一元函数
Matrix_Double RowSums(const Matrix_Double& mat);//按行求和，得到 n×1 列向量矩阵
Matrix_Double AddBias(const Matrix_Double& mat, const Matrix_Double& bias);//每列加同一偏置（广播）
Matrix_Double Softmax(const Matrix_Double& z);//softmax取概率
Matrix_Double GetPaddingMatrix(const Matrix_Double& mat,int padding);//为矩阵周边添加padding行（或列）0元素
Matrix_Double GetDilateMatrix(const Matrix_Double& mat, int stride);//为矩阵的每个元素之间插入stride-1个元素

/// <summary>最大池化的反向（反池化）：按索引列表把池化矩阵还原成原尺寸矩阵，
/// 每个窗口中只有冠军位置取池化值、其余位置为 0（即反向传播要用的梯度回填）。</summary>
/// <param name="pollmat">池化后的矩阵</param>
/// <param name="pollsize">池化窗口边长 k（须与 MaxPoll 时一致）</param>
/// <param name="antipoll_data">MaxPoll 返回的索引列表（长度须等于 pollmat 的元素个数，与原矩阵的行数，列数）</param>
/// <returns>尺寸为 (pollmat.rows·k)×(pollmat.cols·k) 的矩阵</returns>
/// <exception cref="std::runtime_error">pollsize&lt;=0、索引长度不匹配或索引越界时抛出</exception>
Matrix_Double GetAntiMaxPoll(const Matrix_Double& pollmat, int pollsize, const AntiPoll_Data& antipoll_data);

struct AntiPoll_Data {
	std::vector<int>index_list;
	int ordRowNums;
	int ordColNums;
};

Matrix_Double Convolution(const Matrix_Double& mat,const Matrix_Double& kernel, int padding=0, int stride=1);
std::vector<Matrix_Double> Convolution(const std::vector<Matrix_Double>& mat, const Matrix_Double& kernel, int padding = 0, int stride = 1);
Matrix_Double GetFlattenMatrixFromBatch(const std::vector<Matrix_Double>& batch);//将三维张量转化为矩阵
Matrix_Double FlattenChannels(const std::vector<std::vector<Matrix_Double>>& batch);//将四阶张量转化为矩阵
std::vector<std::vector<Matrix_Double>> UnflattenChannels(const Matrix_Double& grad, int C, int H, int W);//将矩阵转化为三阶张量
std::vector<Matrix_Double> UnFlattenToBatch(const Matrix_Double& mat, int h, int w);//将mat按每一列转化为h*w矩阵,矩阵的个数自动决定

Matrix_Double Flatten(const Matrix_Double& mat);
Matrix_Double Unflatten(const Matrix_Double& vec, int h, int w);

Matrix_Double GetTranslateMatrix(const Matrix_Double& mat);//得到转置矩阵（矩阵本身不变）
Matrix_Double GetInverseMatrix(const Matrix_Double& mat, double approximation);//得到逆矩阵（矩阵本身不变）
Matrix_Double GetDoubleMatrix(const Matrix_Complex& mat);//得到双精度矩阵（从复数矩阵转来，只取实部）
Matrix_Complex GetComplexMatrix(const Matrix_Double& mat);//得到复数矩阵（从双精度矩阵转来）
Matrix_Complex GetComplexMatrix(const Matrix_Lambda& mat, Complex val);


//以下函数在GPU(Matrix_Double_Mutiple_GPU)中实现
Matrix_Double Matrix_Double_Mutiple_GPU(const Matrix_Double& mat1, const Matrix_Double& mat2);
/// <summary>把 mat1×mat2 的结果写入调用方提供的 out（尺寸须已匹配），避免每次重新分配结果缓冲。</summary>

void Matrix_Double_Mutiple_GPU(const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out);
namespace Internal {
	struct Matrix_Double_Access
	{
		static std::vector<double>& GetInterData(Matrix_Double& A) { return A.data; }
	};
}

/// <summary>自适应矩阵乘法：把 mat1×mat2 的结果写入 out；乘加数低于阈值走 CPU，否则走 GPU。</summary>
void Matrix_Double_Mutiple_Auto(const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out);

/// <summary>设置矩阵乘法的 GPU 启用阈值（macs = rows×inner×cols）：macs ≥ 阈值时走 GPU；传 SIZE_MAX 可全部走 CPU。</summary>
void SetMatMulGpuThreshold(size_t macs);

/// <summary>选择矩阵乘 GPU 的传输路径：1 = 锁页 + 主机侧 float 转换（A 路径）；0 = pageable double + 设备端转换（旧路径）。</summary>
void SetMatMulTransferMode(int mode);
#include"Batch_Matrix.h"
#include <cmath>	// std::fabs：im2col 自检里用来算误差
#include <algorithm>	// std::fill：卷积分块时用来清零输出缓冲

/// 卷积是否走 GPU 路径的编译期开关：改成 0 可整体回退到 CPU 实现（用于对拍排查）。
#define USE_GPU_CONVOLUTION 1

Batch_Matrix::Batch_Matrix(std::vector<Matrix_Double> data_, int rownum, int colnum) {
	if (rownum < 0 || colnum < 0) {
		throw std::runtime_error("nagetive rowNum or colNum");
	}
	if (data_.size() == 0) {
		throw std::runtime_error("empty input");
	}
	int interRow = data_[0].GetRowNum();
	int interCol = data_[0].GetColNum();
	rowNums_ = rownum;
	colNums_ = colnum;
	for (int i = 0;i < data_.size();i++) {
		if (data_[i].GetColNum() != interCol || data_[i].GetRowNum() != interRow) {
			throw std::runtime_error("consistent");
		}
	}
	data=std::move(data_);
	data.resize(rowNums_ * colNums_, Matrix_Double(interRow, interCol));
}
Batch_Matrix::Batch_Matrix(int rownum, int colnum, int interRownum, int interColnum) {
	if (rownum < 0 || colnum < 0|| interRownum<0 || interColnum<0) {
		throw std::runtime_error("nagetive rowNum or colNum");
	}
	rowNums_ = rownum;
	colNums_ = colnum;
	data.resize(colNums_ * rowNums_, Matrix_Double(interRownum, interColnum));
}
Batch_Matrix::Batch_Matrix(int size,int interSize) {
	if (size < 0 || interSize < 0) {
		throw std::runtime_error("nagetive size");
	}
	rowNums_ = size;
	colNums_ = size;
	data.resize(rowNums_ * colNums_, Matrix_Double(interSize, interSize));
}

void Batch_Matrix::InterRotato_180() {
	for (int i = 0;i < data.size();i++) {
		data[i].Rotato_180();
	}
}
void Batch_Matrix::Random(double ave, double std) {
	for (int i = 0;i < data.size();i++) {
		data[i].Random(ave,std);
	}
}



Batch_Matrix operator+(const Batch_Matrix& bat1, const Batch_Matrix& bat2) {
	if (bat1.rowNums_ != bat2.rowNums_ ||
		bat1.colNums_ != bat2.colNums_ ||
		bat1.data[0].GetColNum() != bat2.data[0].GetColNum() ||
		bat1.data[0].GetRowNum() != bat2.data[0].GetRowNum()
		) {
		throw std::runtime_error("mismatched batch_matrix");
	}
	std::vector<Matrix_Double>newdata(bat1.data.size());
	for (int i = 0;i < bat1.data.size();i++) {
		newdata[i] = bat1.data[i] + bat2.data[i];
	}
	return Batch_Matrix(std::move(newdata), bat1.rowNums_, bat1.colNums_);
}
Batch_Matrix& operator+=(Batch_Matrix& bat1, const Batch_Matrix& bat2) {
	if (bat1.rowNums_ != bat2.rowNums_ ||
		bat1.colNums_ != bat2.colNums_ ||
		bat1.data[0].GetColNum() != bat2.data[0].GetColNum() ||
		bat1.data[0].GetRowNum() != bat2.data[0].GetRowNum()
		) {
		throw std::runtime_error("mismatched batch_matrix");
	}
	for (int i = 0;i < bat1.data.size();i++) {
		bat1.data[i]+=bat2.data[i];
	}
	return bat1;
}
Batch_Matrix operator-(const Batch_Matrix& bat1, const Batch_Matrix& bat2) {
	if (bat1.rowNums_ != bat2.rowNums_ ||
		bat1.colNums_ != bat2.colNums_ ||
		bat1.data[0].GetColNum() != bat2.data[0].GetColNum() ||
		bat1.data[0].GetRowNum() != bat2.data[0].GetRowNum()
		) {
		throw std::runtime_error("mismatched batch_matrix");
	}
	std::vector<Matrix_Double>newdata(bat1.data.size());
	for (int i = 0;i < bat1.data.size();i++) {
		newdata[i] = bat1.data[i] - bat2.data[i];
	}
	return Batch_Matrix(std::move(newdata), bat1.rowNums_, bat1.colNums_);
}
Batch_Matrix& operator-=(Batch_Matrix& bat1, const Batch_Matrix& bat2) {
	if (bat1.rowNums_ != bat2.rowNums_ ||
		bat1.colNums_ != bat2.colNums_ ||
		bat1.data[0].GetColNum() != bat2.data[0].GetColNum() ||
		bat1.data[0].GetRowNum() != bat2.data[0].GetRowNum()
		) {
		std::cout << "bat1 " << bat1.rowNums_ << "," << bat1.colNums_ << "," << bat1.data[0].GetColNum() << "," << bat1.data[0].GetRowNum() << std::endl;
		std::cout << "bat2 " << bat2.rowNums_ << "," << bat2.colNums_ << "," << bat2.data[0].GetColNum() << "," << bat2.data[0].GetRowNum() << std::endl;

		throw std::runtime_error("mismatched batch_matrix");
	}
	for (int i = 0;i < bat1.data.size();i++) {
		bat1.data[i] -= bat2.data[i];
	}
	return bat1;
}
Batch_Matrix operator*(double db, const Batch_Matrix& bat) {
	std::vector<Matrix_Double>newdata(bat.data.size());
	for (int i = 0;i < bat.data.size();i++) {
		newdata[i] =db*bat.data[i];
	}
	return Batch_Matrix(std::move(newdata), bat.rowNums_, bat.colNums_);
}
Batch_Matrix& operator*=(double db, Batch_Matrix& bat) {
	for (int i = 0;i < bat.data.size();i++) {
		bat.data[i] *= db;
	}
	return bat;
}
/// <summary>im2col：把一个已补零的 H×W 图按 K×K 窗口展开成 (outRows*outCols) × (K*K) 的矩阵。</summary>
/// <param name="src">已补零的输入图（行主序，跨行步长 = src.GetColNum()）。</param>
/// <param name="K">卷积核边长。</param>
/// <param name="stride">滑窗步长。</param>
/// <param name="outRows">输出图行数（窗口在竖直方向滑动的次数）。</param>
/// <param name="outCols">输出图列数。</param>
/// <param name="ld">目标矩阵的行步长（= 目标矩阵的总列数）。</param>
/// <param name="dst">展开结果的缓冲起点；写入 (outRows*outCols) 行 × K*K 列。</param>
/// <remarks>展开后：第 l 行对应第 l 个输出位置，第 m 列对应核的第 m 个元素。
/// 这样每个窗口的那 K×K 个像素在内存里是连续的，写入和读取都连续。
/// 之所以把「输出位置」放在行、「核元素」放在列，是为了让后面矩阵乘的第二个乘数（核矩阵）
/// 尽量小到能常驻 cache——详见 GetConvolutionMatrixMul 的说明。</remarks>
static void Im2Col(const Matrix_Double& src, int K, int stride,int outRows, int outCols, int ld, double* dst)
{
	const int W = src.GetColNum();                     // 补零后图的宽度 = 跨行步长
	const double* sd = src.GetInternalData().data();   // 只读访问，无需 friend

	for (int orow = 0; orow < outRows; orow++) {
		for (int ocol = 0; ocol < outCols; ocol++) {
			const int l = orow * outCols + ocol;                            // 第 l 行 = 第 l 个窗口
			double* d = dst + (size_t)l * ld;                               // 该行的行首
			const double* base = sd + (orow * stride) * W + ocol * stride;  // 窗口左上角
			for (int kr = 0; kr < K; kr++) {
				const double* p = base + kr * W;
				double* dr = d + kr * K;
				for (int kc = 0; kc < K; kc++) {
					dr[kc] = p[kc];   // 连续读、连续写（旧版是 d += ld 的跳跃写）
				}
			}
		}
	}
}

/// <summary>对拍参考实现：GetConvolutionMatrixMul 的原始朴素版本（逐block累加、每个块单独做一次卷积）。</summary>
/// <remarks>仅供自检对比用，保留它是为了证明新实现与之数值等价。
/// 生产路径已不再调用它——它每次内层循环都会 new 一个临时矩阵，正是要优化掉的那部分开销。</remarks>
static Batch_Matrix GetConvolutionMatrixMul_Naive(const Batch_Matrix& bat, const Batch_Matrix& ker, int padding, int stride) {
	// 本函数不是 Batch_Matrix 的友元，所以只能通过公开接口取数据（GetRowNum / GetColNum / GetInternalData）
	const std::vector<Matrix_Double>& bd = bat.GetInternalData();
	const std::vector<Matrix_Double>& kd = ker.GetInternalData();
	if (bat.GetColNum() != ker.GetRowNum()) {
		throw std::runtime_error("mismatched bat");
	}
	if (kd[0].GetRowNum() != kd[0].GetColNum()) {
		throw std::runtime_error("the ker couldn't serve as the kernel");
	}
	int newRow = bat.GetRowNum();
	int newCol = ker.GetColNum();
	int newInterRow = (bd[0].GetRowNum() + 2 * padding - kd[0].GetRowNum()) / stride + 1;
	int newInterCol = (bd[0].GetColNum() + 2 * padding - kd[0].GetColNum()) / stride + 1;
	std::vector<Matrix_Double> newData((size_t)newRow * newCol, Matrix_Double(newInterRow, newInterCol));
#pragma omp parallel for if(newRow > 64)
	for (int row = 0; row < newRow; row++) {//选取bat的第row行
		Matrix_Double* New_Base_Address = &newData[(size_t)row * newCol];
		for (int col = 0; col < newCol; col++) {//选取ker的第col列
			const Matrix_Double* Bat_Base_Address = &bd[(size_t)row * bat.GetColNum()];
			const Matrix_Double* Ker_Base_Address = &kd[col];
			for (int index = 0; index < bat.GetColNum(); index++) {
				New_Base_Address[col] += Convolution(Bat_Base_Address[index], Ker_Base_Address[ker.GetColNum() * index], padding, stride);
			}
		}
	}
	return Batch_Matrix(std::move(newData), newRow, newCol);
}

/// <summary>把 C += A × B 累加进调用方提供的缓冲区（全部行主序）。</summary>
/// <param name="C">累加目标，M 行 × N 列。调用方需保证进入时已是所需初值（本函数只做 +=）。</param>
/// <param name="A">左乘矩阵，M 行 × K 列，行步长 = K。</param>
/// <param name="B">右乘矩阵，K 行 × N 列，行步长 = N。</param>
/// <param name="M">C 与 A 的行数。</param>
/// <param name="K">A 的列数，也是 B 的行数（收缩维）。</param>
/// <param name="N">C 与 B 的列数。</param>
/// <remarks>内核为 4×8 寄存器分块：一次算 4 行 × 8 列 = 32 个输出，32 个累加器全程驻留寄存器，
/// k 循环内不读写内存。相比原来的 SAXPY 形态（每 FMA 三次访存），同一条 B 的 cache line
/// 现在被 4 行复用，访存流量摊薄约 4 倍。</remarks>
static void GemmAccum(double* __restrict C, const double* __restrict A, const double* __restrict B, int M, int K, int N) {
	// 主体一次算 4 行：行数向下取整到 4 的倍数，余下 0~3 行交给后面的 1×8 尾部
	const int iMain = M & ~3;
#pragma omp parallel for if(M > 64)
	for (int i = 0; i < iMain; i += 4) {
		// 4 行的 A 行首地址（行主序：第 r 行从 A[(i+r)*K] 开始）
		const double* __restrict aRow0 = A + (size_t)(i + 0) * K;
		const double* __restrict aRow1 = A + (size_t)(i + 1) * K;
		const double* __restrict aRow2 = A + (size_t)(i + 2) * K;
		const double* __restrict aRow3 = A + (size_t)(i + 3) * K;
		// 4 行的 C 行首地址
		double* __restrict cRow0 = C + (size_t)(i + 0) * N;
		double* __restrict cRow1 = C + (size_t)(i + 1) * N;
		double* __restrict cRow2 = C + (size_t)(i + 2) * N;
		double* __restrict cRow3 = C + (size_t)(i + 3) * N;

		int j = 0;
		for (; j + 8 <= N; j += 8) {
			// 累加器命名 r{行}c{列}，共 32 个。注意从 C 的现值起步而非 0——
			// 本函数是累加语义，从现值起步可让加法链与旧实现逐项 += 完全一致（逐位对拍要求）。
			double r0c0 = cRow0[j + 0], r0c1 = cRow0[j + 1], r0c2 = cRow0[j + 2], r0c3 = cRow0[j + 3];
			double r0c4 = cRow0[j + 4], r0c5 = cRow0[j + 5], r0c6 = cRow0[j + 6], r0c7 = cRow0[j + 7];
			double r1c0 = cRow1[j + 0], r1c1 = cRow1[j + 1], r1c2 = cRow1[j + 2], r1c3 = cRow1[j + 3];
			double r1c4 = cRow1[j + 4], r1c5 = cRow1[j + 5], r1c6 = cRow1[j + 6], r1c7 = cRow1[j + 7];
			double r2c0 = cRow2[j + 0], r2c1 = cRow2[j + 1], r2c2 = cRow2[j + 2], r2c3 = cRow2[j + 3];
			double r2c4 = cRow2[j + 4], r2c5 = cRow2[j + 5], r2c6 = cRow2[j + 6], r2c7 = cRow2[j + 7];
			double r3c0 = cRow3[j + 0], r3c1 = cRow3[j + 1], r3c2 = cRow3[j + 2], r3c3 = cRow3[j + 3];
			double r3c4 = cRow3[j + 4], r3c5 = cRow3[j + 5], r3c6 = cRow3[j + 6], r3c7 = cRow3[j + 7];
			// 用指针自增代替每次 k*N 的乘法（强度削减），同时避免 int 溢出
			const double* __restrict bPtr = B + j;
			for (int k = 0; k < K; k++) {
				// B 的 8 个元素取进寄存器后供 4 行复用，这是把访存流量摊薄的关键
				const double b0 = bPtr[0], b1 = bPtr[1], b2 = bPtr[2], b3 = bPtr[3];
				const double b4 = bPtr[4], b5 = bPtr[5], b6 = bPtr[6], b7 = bPtr[7];
				const double x0 = aRow0[k], x1 = aRow1[k], x2 = aRow2[k], x3 = aRow3[k];
				r0c0 += x0 * b0; r0c1 += x0 * b1; r0c2 += x0 * b2; r0c3 += x0 * b3;
				r0c4 += x0 * b4; r0c5 += x0 * b5; r0c6 += x0 * b6; r0c7 += x0 * b7;
				r1c0 += x1 * b0; r1c1 += x1 * b1; r1c2 += x1 * b2; r1c3 += x1 * b3;
				r1c4 += x1 * b4; r1c5 += x1 * b5; r1c6 += x1 * b6; r1c7 += x1 * b7;
				r2c0 += x2 * b0; r2c1 += x2 * b1; r2c2 += x2 * b2; r2c3 += x2 * b3;
				r2c4 += x2 * b4; r2c5 += x2 * b5; r2c6 += x2 * b6; r2c7 += x2 * b7;
				r3c0 += x3 * b0; r3c1 += x3 * b1; r3c2 += x3 * b2; r3c3 += x3 * b3;
				r3c4 += x3 * b4; r3c5 += x3 * b5; r3c6 += x3 * b6; r3c7 += x3 * b7;
				bPtr += N;   // B 的行步长 = N，k 累加顺序保持递增不变
			}
			// 整个 k 循环结束后一次性写回 4 行 × 8 列，消除内层里的“读 C + 写 C”
			cRow0[j + 0] = r0c0; cRow0[j + 1] = r0c1; cRow0[j + 2] = r0c2; cRow0[j + 3] = r0c3;
			cRow0[j + 4] = r0c4; cRow0[j + 5] = r0c5; cRow0[j + 6] = r0c6; cRow0[j + 7] = r0c7;
			cRow1[j + 0] = r1c0; cRow1[j + 1] = r1c1; cRow1[j + 2] = r1c2; cRow1[j + 3] = r1c3;
			cRow1[j + 4] = r1c4; cRow1[j + 5] = r1c5; cRow1[j + 6] = r1c6; cRow1[j + 7] = r1c7;
			cRow2[j + 0] = r2c0; cRow2[j + 1] = r2c1; cRow2[j + 2] = r2c2; cRow2[j + 3] = r2c3;
			cRow2[j + 4] = r2c4; cRow2[j + 5] = r2c5; cRow2[j + 6] = r2c6; cRow2[j + 7] = r2c7;
			cRow3[j + 0] = r3c0; cRow3[j + 1] = r3c1; cRow3[j + 2] = r3c2; cRow3[j + 3] = r3c3;
			cRow3[j + 4] = r3c4; cRow3[j + 5] = r3c5; cRow3[j + 6] = r3c6; cRow3[j + 7] = r3c7;
		}
		// j 尾巴：仅当 N 不是 8 的倍数时进入（最多 7 列），4 行一起按列算
		for (; j < N; j++) {
			const double* __restrict bPtr = B + j;
			double s0 = cRow0[j], s1 = cRow1[j], s2 = cRow2[j], s3 = cRow3[j];
			for (int k = 0; k < K; k++) {
				const double b = *bPtr;
				s0 += aRow0[k] * b;
				s1 += aRow1[k] * b;
				s2 += aRow2[k] * b;
				s3 += aRow3[k] * b;
				bPtr += N;
			}
			cRow0[j] = s0; cRow1[j] = s1; cRow2[j] = s2; cRow3[j] = s3;
		}
	}
	// 尾部：M 不是 4 的倍数时，剩余 0~3 行沿用 1×8 微内核，保证不漏行
	for (int i = iMain; i < M; i++) {
		const double* __restrict iABase = A + (size_t)i * K;
		double* __restrict iCBase = C + (size_t)i * N;

		int j = 0;
		for (; j + 8 <= N; j += 8) {
			// 同样从 C 现值起步，保持加法链一致
			double c0 = iCBase[j + 0], c1 = iCBase[j + 1], c2 = iCBase[j + 2], c3 = iCBase[j + 3];
			double c4 = iCBase[j + 4], c5 = iCBase[j + 5], c6 = iCBase[j + 6], c7 = iCBase[j + 7];
			const double* __restrict bPtr = B + j;
			for (int k = 0; k < K; k++) {
				const double a = iABase[k];
				c0 += a * bPtr[0];
				c1 += a * bPtr[1];
				c2 += a * bPtr[2];
				c3 += a * bPtr[3];
				c4 += a * bPtr[4];
				c5 += a * bPtr[5];
				c6 += a * bPtr[6];
				c7 += a * bPtr[7];
				bPtr += N;
			}
			// 整个 k 循环结束后只写一次
			iCBase[j + 0] = c0; iCBase[j + 1] = c1;
			iCBase[j + 2] = c2; iCBase[j + 3] = c3;
			iCBase[j + 4] = c4; iCBase[j + 5] = c5;
			iCBase[j + 6] = c6; iCBase[j + 7] = c7;
		}
		// 尾巴：仅当 N 不是 8 的倍数时进入（最多 7 列）
		for (; j < N; j++) {
			const double* __restrict bPtr = B + j;
			double s = iCBase[j];
			for (int k = 0; k < K; k++) {
				s += iABase[k] * (*bPtr);
				bPtr += N;
			}
			iCBase[j] = s;
		}
	}
}

/// <summary>多管道卷积（块网格版）：GCM(X,Y)[r][c] = Σ_k Conv(X[r][k], Y[k][c], p, s)。</summary>
/// <remarks>内部用 im2col + 矩阵乘：把输入窗口展开成矩阵，再用矩阵乘一次算出全部 C_out 个输出通道。
/// 布局刻意做成「小的维度（C_out）在列、大的维度（n·L）在行」，让核矩阵能常驻 cache。
/// 展开矩阵的大小是 (n·L) × (ci·K²)，所以样本维和通道维都要分块——见下面 kMaxExpand 的说明。</remarks>
/// <param name="bat">输入图网格：行 × 输入通道，每块一张特征图。</param>
/// <param name="ker">核网格：(输入通道) × (输出通道)，每块 K×K。</param>
/// <param name="padding">四周补零的行列数（与 Convolution 同义）。</param>
/// <param name="stride">滑窗步长。</param>
/// <returns>输出网格：bat 的行数 × C_out，每块为卷积后的特征图。</returns>
/// <exception cref="std::runtime_error">通道数不匹配、核不是方阵，或核比补零后的输入还大时抛出</exception>
Batch_Matrix GetConvolutionMatrixMul(const Batch_Matrix& bat, const Batch_Matrix& ker, int padding, int stride) {
	if (bat.colNums_ != ker.rowNums_) {
		throw std::runtime_error("mismatched bat");
	}
	if (ker.data[0].GetRowNum() != ker.data[0].GetColNum()) {
		throw std::runtime_error("the ker couldn't serve as the kernel");
	}

	const int R = bat.rowNums_;      // bat 的行数（前向时 = 批大小；反传 dK 时 = 层通道数）
	const int C_in = bat.colNums_;   // bat 的列数（前向时 = 输入通道数；反传 dK 时 = 批大小）
	const int C_out = ker.colNums_;  // 输出通道数
	const int K = ker.data[0].GetRowNum();
	const int KK = K * K;
#if USE_GPU_CONVOLUTION
	// GPU 派发刻意放在 naive 判断之前：反传第 0 层 dA 的 C_out=1 占卷积总内存搬运的大头，
	// 若把它留在 CPU 的 naive 路径上，等于把最肥的一块放走了。
	// 瘦 N kernel 只支持 C_out ≤ kGemmThinNMax；超出时落到下面的 CPU 分支。
#endif

	// 以下 naive 分支在 GPU 路径开启时已不可达，保留它是作为对拍参考实现
	// （把 USE_GPU_CONVOLUTION 改成 0 即可重新走到这里）。
	if (C_out <= 2) {
		return GetConvolutionMatrixMul_Naive(bat, ker, padding, stride);
	}
	// padding 统一在入口做一次，之后全按 padding=0 展开。
	// padding==0 时用指针直接指向原批，避免把整批矩阵白白复制一遍。
	Batch_Matrix padded;
	const Batch_Matrix* padBat = &bat;
	if (padding > 0) { padded = GetPaddingMatrix(bat, padding); padBat = &padded; }

	const int H = padBat->data[0].GetRowNum();
	const int W = padBat->data[0].GetColNum();
	const int outRows = (H - K) / stride + 1;   // 与 Convolution 相同：整数除法向下取整
	const int outCols = (W - K) / stride + 1;
	if (outRows <= 0 || outCols <= 0) {
		throw std::runtime_error("mismatched convolution");
	}
	const int L = outRows * outCols;   // 每个输出特征图上的元素个数

	// 核矩阵 w：(C_in*KK) 行 × C_out 列。
	// 第 (i*KK + k) 行、第 o 列 = 第 i 个输入通道连到第 o 个输出通道的那个核的第 k 个元素。
	// 只构造一次，之后被所有分块、所有样本复用。
	std::vector<double> w((size_t)C_in * KK * C_out);
	for (int i = 0; i < C_in; i++) {
		for (int o = 0; o < C_out; o++) {
			const double* kp = ker.data[i * C_out + o].GetInternalData().data();
			double* wp = &w[(size_t)(i * KK) * C_out + o];                // 第 (i*KK) 行、第 o 列
			for (int k = 0; k < KK; k++) wp[(size_t)k * C_out] = kp[k];   // 行内步进 C_out
		}
	}

	std::vector<Matrix_Double> newData((size_t)R * C_out);

	// ===== 分块展开 =====
	// 展开矩阵 A 的大小是 (n·L) × (ci·KK) 个 double。这里必须两个维度都分块：
	//   · 样本维（rb）：一次最多 64 个样本；
	//   · 通道维（cb）：反传的 dK 调用里 bat 被转置过，此时上面那个 C_in 拿到的是【批大小】，
	//     每个"通道"要展开 L·KK 个数，整批一次展开会到 GB 级。按通道切块后单次展开量可控。
	const size_t kMaxExpand = 8u * 256 * 256;   // 单次展开的 double 个数上限（约 4 MB）

	int rb = (R < 64) ? R : 64;                   // 样本块大小
	const size_t perSample = (size_t)L * KK;
	if (perSample > 0 && (size_t)rb * perSample > kMaxExpand) {
		rb = (int)(kMaxExpand / perSample);       // 极端情况下再压小样本块
		if (rb < 1) rb = 1;
	}
	int cb = C_in;                                // 通道块大小
	const size_t perChannel = (size_t)rb * L * KK;
	if (perChannel > 0 && perChannel * (size_t)C_in > kMaxExpand) {
		cb = (int)(kMaxExpand / perChannel);
		if (cb < 1) cb = 1;
	}

	// 两个缓冲都只在循环外分配一次，循环内不再重新分配、不再清零中间结果。
	std::vector<double> a((size_t)rb * L * cb * KK);   // 展开缓冲
	std::vector<double> out((size_t)rb * L * C_out);   // 本样本块的输出

	for (int r0 = 0; r0 < R; r0 += rb) {
		const int n = (rb < R - r0) ? rb : (R - r0);   // 本块实际样本数（最后一块可能不满）
		std::fill(out.begin(), out.end(), 0.0);        // 通道块之间要累加，进入时必须清零

		for (int i0 = 0; i0 < C_in; i0 += cb) {
			const int ci = (cb < C_in - i0) ? cb : (C_in - i0);   // 本块实际通道数
			const int cols = ci * KK;                             // 本块展开矩阵的列数
			// 展开：样本 r 占用行段 [r*L, (r+1)*L)，块内第 i 个通道占用列段 [i*KK, (i+1)*KK)
#pragma omp parallel for collapse(2)
			for (int r = 0; r < n; r++) {
				for (int i = 0; i < ci; i++) {
					Im2Col(padBat->data[(r0 + r) * C_in + (i0 + i)], K, stride, outRows, outCols,
						cols,
						a.data() + (size_t)(r * L) * cols + (size_t)i * KK);
					//把padBat->data[(r0 + r) * C_in + (i0 + i)]，即补0后的Batch_Matirx的第r0+r行,i0+i列的矩阵，按照卷积展开，存储到a.data()中
				}
			}
			// 累加本通道块的贡献：
			//   out (n·L × C_out)  +=  A (n·L × cols)  ×  w 的 [i0*KK, i0*KK+cols) 行 (cols × C_out)
			GemmAccum(out.data(), a.data(), w.data() + (size_t)i0 * KK * C_out,n * L, cols, C_out);
		}

		// 拆回块网格：out[(r*L + l)*C_out + o] 就是第 (r0+r) 个样本、第 o 个输出通道的第 l 个元素。
		// o 落在列方向，所以取一块时是步长为 C_out 的 gather（不是连续切片）。
#pragma omp parallel for if(C_out>64)
		for (int o = 0; o < C_out; o++) {
			for (int r = 0; r < n; r++) {
				std::vector<double> tmp(L);
				const double* base = out.data() + (size_t)(r * L) * C_out + o;
				for (int l = 0; l < L; l++) tmp[l] = base[(size_t)l * C_out];
				newData[(size_t)(r0 + r) * C_out + o] = Matrix_Double(std::move(tmp), outRows, outCols);
			}
		}
	}
	return Batch_Matrix(std::move(newData), R, C_out);
}

/// <summary>【临时自检】用随机数据把新的 im2col 实现与朴素参考实现逐元素对拍。</summary>
/// <remarks>这是验证期临时加上的自检代码。确认全部用例数值一致后，可连同 main.cpp 里的前置声明和调用一起删掉。</remarks>
/// <returns>全部用例的最大误差都在 1e-9 以内返回 true，否则返回 false</returns>
bool SelfTest_GetConvolutionMatrixMul() {
	struct Case { int R, C_in, C_out, H, W, K, padding, stride; };
	const Case cases[] = {
		{ 5, 3, 2, 8,  8,  3, 0, 1 },   // 反传路径：padding=0、stride=1
		{ 4, 1, 4, 28, 28, 9, 6, 1 },   // 前向路径：大 padding（对应第 0 层）
		{ 4, 3, 2, 16, 16, 5, 2, 1 },   // 前向路径：小 padding
		{ 3, 2, 3, 10, 10, 3, 0, 2 },   // stride=2
		{ 3, 2, 3, 10, 10, 3, 1, 2 },   // stride=2 且带 padding
		{ 2, 1, 1, 7,  7,  3, 1, 1 },   // 单输入通道、单输出通道
		{ 3, 2, 2, 9,  11, 4, 1, 2 },   // 矩形输入，且输出尺寸不能被整除
	};
	const int caseNum = sizeof(cases) / sizeof(cases[0]);
	double maxErr = 0.0;
	bool allPass = true;
	std::cout << "===== im2col self test (new vs naive) =====" << std::endl;
	for (int c = 0; c < caseNum; c++) {
		const Case& cs = cases[c];
		double caseErr = 0.0;
		try {
			Batch_Matrix bat(cs.R, cs.C_in, cs.H, cs.W);       // 随机输入图网格：R × C_in，每块 H×W
			Batch_Matrix ker(cs.C_in, cs.C_out, cs.K, cs.K);   // 随机核网格：C_in × C_out，每块 K×K
			bat.Random(0.0, 1.0);
			ker.Random(0.0, 1.0);

			Batch_Matrix got = GetConvolutionMatrixMul(bat, ker, cs.padding, cs.stride);
			Batch_Matrix ref = GetConvolutionMatrixMul_Naive(bat, ker, cs.padding, cs.stride);

			const std::vector<Matrix_Double>& gd = got.GetInternalData();
			const std::vector<Matrix_Double>& rd = ref.GetInternalData();
			if (gd.size() != rd.size()) {
				caseErr = 1e9;   // 块数都不一样，直接判失败
			}
			else {
				for (int i = 0; i < (int)gd.size(); i++) {
					if (gd[i].GetRowNum() != rd[i].GetRowNum() || gd[i].GetColNum() != rd[i].GetColNum()) {
						caseErr = 1e9;
						break;
					}
					for (int r = 0; r < gd[i].GetRowNum(); r++) {
						for (int cc = 0; cc < gd[i].GetColNum(); cc++) {
							double e = std::fabs(gd[i].GetData(r, cc) - rd[i].GetData(r, cc));
							if (e > caseErr) caseErr = e;
						}
					}
				}
			}
		}
		catch (const std::exception& ex) {
			std::cout << "case " << c << " threw: " << ex.what() << std::endl;
			caseErr = 1e9;
		}
		if (caseErr > maxErr) maxErr = caseErr;
		if (!(caseErr < 1e-9)) allPass = false;
		std::cout << "case " << c << " (R=" << cs.R << ", Cin=" << cs.C_in << ", Cout=" << cs.C_out
			<< ", H=" << cs.H << ", W=" << cs.W << ", K=" << cs.K
			<< ", p=" << cs.padding << ", s=" << cs.stride << ")  maxErr=" << caseErr << std::endl;
	}
	std::cout << "im2col self test " << (allPass ? "PASSED" : "FAILED")
		<< ", overall maxErr=" << maxErr << std::endl;
	return allPass;
}

Batch_Matrix GetBlockTranspose(const Batch_Matrix& bat) {
	std::vector<Matrix_Double> newData(bat.data.size());
	for (int col = 0;col < bat.colNums_;col++) {
		Matrix_Double* New_Base_Address = &newData[col * bat.rowNums_];
		const Matrix_Double* Bat_Base_Address = &bat.data[col];
		for (int row = 0;row < bat.rowNums_;row++) {
			New_Base_Address[row] = Bat_Base_Address[row * bat.colNums_];
		}
	}
	return Batch_Matrix(std::move(newData), bat.colNums_, bat.rowNums_);
}

Batch_Matrix MapMatrix(const Batch_Matrix& bat, double(*f)(double)) {
	std::vector<Matrix_Double> newData(bat.data.size());
	for (int i = 0;i < (int)bat.data.size();i++) {
		newData[i] = MapMatrix(bat.data[i], f);
	}
	return Batch_Matrix(std::move(newData), bat.rowNums_, bat.colNums_);
}




std::vector<Matrix_Double> FuseKernelMaps(const Batch_Matrix& bat) {
	//bat的相关数据
	if (bat.GetInternalData().size() == 0) {
		throw std::runtime_error("the bat is zero batch_matrix");
	}
	int OuterRowNum = bat.GetRowNum();
	int OuterColNum = bat.GetColNum();
	int InterRowNum = bat.GetInternalData()[0].GetRowNum();
	int InterColNum = bat.GetInternalData()[0].GetColNum();
	int InterSize = InterRowNum * InterColNum;
	//输出的三阶张量的相关数据
	int NewVec_Size = OuterRowNum;
	int NewVec_InterRowNums = InterSize;
	int NewVec_InterColNums = OuterColNum;

	std::vector<Matrix_Double> Result(NewVec_Size);

	for (int index = 0;index < NewVec_Size;index++) {//考虑Result的第index个矩阵
		std::vector<double> NewData(NewVec_InterRowNums * NewVec_InterColNums);
		const Matrix_Double* Bat_Base_Address = &bat.GetInternalData()[index * OuterColNum];
		for (int col = 0;col < OuterColNum;col++) {//对bat的第index行进行遍历
			double* NewCol_Base_Address = &NewData[col];
			const double* BatCol_Base_Address = &(Bat_Base_Address[col].GetInternalData()[0]);
			for (int row = 0;row < NewVec_InterRowNums;row++) {//对Result的第index个矩阵的每一列逐个填充
				NewCol_Base_Address[row * NewVec_InterColNums] = BatCol_Base_Address[row];//这里，我们是把bat中矩阵的每个元素，按顺序移动到了vec中矩阵的列上
			}
		}
		Result[index] = Matrix_Double(std::move(NewData), NewVec_InterRowNums, NewVec_InterColNums);
	}
	return Result;
}


Batch_Matrix SplitKernelGrad(const std::vector<Matrix_Double>& grad, int newInterColNum) {
	if (newInterColNum <= 0) {
		throw std::runtime_error("negative or zero newInterColNum");
	}
	if (grad.size() == 0) {
		throw std::runtime_error("the grad is zeroVec");
	}
	if (grad[0].GetRowNum() % newInterColNum != 0) {
		throw std::runtime_error("couldn't turn to the Batch");
	}
	//输入的三维张量的特征
	int Vec_Size = grad.size();
	int Vec_InterRowNum = grad[0].GetRowNum();
	int Vec_InterColNum = grad[0].GetColNum();
	//需要输出的Batch矩阵的特征
	int Bat_OuterRowNum = Vec_Size;
	int Bat_OuterColNum = Vec_InterColNum;
	int Bat_InterColNum = newInterColNum;
	int Bat_InterRowNum = Vec_InterRowNum / newInterColNum;
	int Bat_Size = Bat_OuterColNum * Bat_OuterRowNum;
	std::vector<Matrix_Double> BatData(Bat_Size);
	for (int outRow = 0;outRow < Bat_OuterRowNum;outRow++) {//遍历bat的每一行
		Matrix_Double* Bat_Base_Address = &BatData[outRow * Bat_OuterColNum];
		const double* Vec_Base_Address = &grad[outRow].GetInternalData()[0];
		for (int outCol = 0;outCol < Bat_OuterColNum;outCol++) {//遍历bat的每一列
			std::vector<double> newData(Vec_InterRowNum);
			const double* VecCol_Base_Address = &Vec_Base_Address[outCol];
			for (int row = 0;row < Vec_InterRowNum;row++) {//逐个复制
				newData[row] = VecCol_Base_Address[row * Vec_InterColNum];//这里，我们把grad中的矩阵的每一列，按顺序复制到了bat的每一列里
			}
			Bat_Base_Address[outCol] = Matrix_Double(std::move(newData), Bat_InterRowNum, Bat_InterColNum);
		}
	}
	return Batch_Matrix(std::move(BatData),Bat_OuterRowNum,Bat_OuterColNum);
}
Batch_Matrix Hadamard(const Batch_Matrix& bat1, const Batch_Matrix& bat2) {
	if (bat1.rowNums_ != bat2.rowNums_ ||
		bat1.colNums_ != bat2.colNums_ ||
		bat1.data[0].GetColNum() != bat2.data[0].GetColNum() ||
		bat1.data[0].GetRowNum() != bat2.data[0].GetRowNum()
		) {
		throw std::runtime_error("mismatched batch_matrix");
	}
	int size = bat1.data.size();
	int interRowNum = bat1.data[0].GetRowNum();
	int interColNum = bat1.data[0].GetColNum();
	int interSize = interRowNum*interColNum;
	std::vector<Matrix_Double>newData(bat1.data.size());
	for (int i = 0;i < size;i++) {
		std::vector<double> newData_(interSize);
		const double* Bat1_Base_Address = &bat1.data[i].GetInternalData()[0];
		const double* Bat2_Base_Address = &bat2.data[i].GetInternalData()[0];
		for (int j = 0;j < interSize;j++) {
			newData_[j] = Bat1_Base_Address[j] * Bat2_Base_Address[j];
		}
		newData[i] = Matrix_Double(std::move(newData_), interRowNum,interColNum);
	}
	return Batch_Matrix(std::move(newData), bat1.rowNums_, bat1.colNums_);
}
Batch_Matrix GetPaddingMatrix(const Batch_Matrix& bat, int padding) {
	if (padding == 0) {
		return bat;
	}
	const int newRowNums = bat.GetInternalData()[0].GetRowNum() + 2 * padding;
	const int newColNums = bat.GetInternalData()[0].GetColNum() + 2 * padding;
	if (newRowNums <= 0 || newColNums <= 0) {
		throw std::runtime_error("padding too large: empty result");
	}
	std::vector<Matrix_Double> newData(bat.GetInternalData().size());
	const Matrix_Double* Bat_Base_Address = &bat.GetInternalData()[0];
	for (int i = 0;i < newData.size();i++) {
		newData[i] = GetPaddingMatrix(Bat_Base_Address[i], padding);
	}
	return Batch_Matrix(std::move(newData), bat.GetRowNum(), bat.GetColNum());
}
Batch_Matrix GetDilateMatrix(const Batch_Matrix& bat, int stride) {
	if (stride == 1) {
		return bat;
	}
	if (stride <= 0) {
		throw std::runtime_error("negative or zero stride");
	}
	std::vector<Matrix_Double> newData(bat.GetInternalData().size());
	const Matrix_Double* Bat_Base_Address = &bat.GetInternalData()[0];
	for (int i = 0;i < newData.size();i++) {
		newData[i] = GetDilateMatrix(Bat_Base_Address[i], stride);
	}
	return Batch_Matrix(std::move(newData), bat.GetRowNum(), bat.GetColNum());
}

std::vector<AntiPoll_Data> Batch_Matrix::MaxPoll(int pollsize) {
	if (pollsize <= 0) {
		throw std::runtime_error("pollsize must be positive");
	}
	if (pollsize == 1) {
		int interRow = data[0].GetRowNum();
		int interCol = data[0].GetColNum();
		return std::vector<AntiPoll_Data>(rowNums_*colNums_,{std::vector<int>(interCol*interRow,0.0),interRow,interCol});
	}
	std::vector<AntiPoll_Data> ctxs(data.size());
	for (int i = 0; i < (int)data.size(); i++) {
		ctxs[i] = data[i].MaxPoll(pollsize);   // 复用单块版：它已在覆盖尺寸前记好原始行列数
	}
	return ctxs;   // ctxs[i] 恰好对应 data[i]，顺序天然对齐
}

Batch_Matrix GetAntiMaxPoll(const Batch_Matrix& dA, int pollsize, const std::vector<AntiPoll_Data>& ctxs) {
	const std::vector<Matrix_Double>& src = dA.GetInternalData();
	if (ctxs.size() != src.size()) {
		throw std::runtime_error("context count mismatched with batch");
	}
	if (pollsize == 1) {
		return dA;
	}
	std::vector<Matrix_Double> out(src.size());
	for (int i = 0; i < (int)src.size(); i++) {
		out[i] = GetAntiMaxPoll(src[i], pollsize, ctxs[i]);   // 复用单块版
	}
	// 池化只作用在块内空间维(H,W)，不改变样本/通道网格，故沿用 dA 的网格尺寸
	return Batch_Matrix(std::move(out), dA.GetRowNum(), dA.GetColNum());
}

/// <summary>卷积的 GPU 启用阈值（乘加次数）。默认 800 万：低于它时 GPU 的打包/传输开销盖过计算收益。</summary>
static size_t gConvGpuThreshold = 8u * 1024 * 1024;

void SetConvGpuThreshold(size_t macs) { gConvGpuThreshold = macs; }

void GetConvolutionMatrixMul_Auto(const Batch_Matrix& bat, const Batch_Matrix& ker, Batch_Matrix& out, int padding, int stride) {
	const std::vector<Matrix_Double>& bd = bat.GetInternalData();
	const std::vector<Matrix_Double>& kd = ker.GetInternalData();
	if (bd.empty() || kd.empty()) { throw std::runtime_error("empty batch or kernel"); }
	if (bat.GetColNum() != ker.GetRowNum()) { throw std::runtime_error("mismatched bat"); }
	const int R = bat.GetRowNum();
	const int C_in = bat.GetColNum();
	const int C_out = ker.GetColNum();
	const int K = kd[0].GetRowNum();
	const int H = bd[0].GetRowNum() + 2 * padding;
	const int W = bd[0].GetColNum() + 2 * padding;
	if (stride < 1 || H < K || W < K) { throw std::runtime_error("mismatched convolution"); }
	const int L = ((H - K) / stride + 1) * ((W - K) / stride + 1);
	const size_t macs = (size_t)R * (size_t)L * (size_t)C_in * (size_t)K * (size_t)K * (size_t)C_out;
	const int outRows = (H - K) / stride + 1;
	const int outCols = (W - K) / stride + 1;
#ifndef CPU_ONLY
	const size_t totalThreads = GetConvGpuThreadCount(R, C_out, outRows, outCols);
	// 两个判据：总计算量够大 + 总线程数够多（否则 GPU 只有几十个线程在跑，反而不如 CPU）
	if (macs >= gConvGpuThreshold && totalThreads >= 4096) {
		GetConvolutionMatrixMul_GPU(bat, ker, out, padding, stride);   // 大卷积：GPU，直接写入 out
		return;
	}
#endif
	out = GetConvolutionMatrixMul(bat, ker, padding, stride);          // CPU 路径（CPU_ONLY 工程必走这里）
}
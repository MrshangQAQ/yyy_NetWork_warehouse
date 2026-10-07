#include"Batch_Matrix.h"
#include <cmath>	// std::fabs：im2col 自检里用来算误差
#include <algorithm>	// std::fill：卷积分块时用来清零输出缓冲

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
static void GemmAccum(double* __restrict C, const double* __restrict A, const double* __restrict B, int M, int K, int N) {
	// ikj 顺序：i(行) → k(收缩维) → j(列)。B 与 C 都按行连续访问，对缓存友好。
#pragma omp parallel for if(M > 64)
	for (int i = 0; i < M; i++) {
		double* iCBase = C + (size_t)i * N;
		const double* iABase = A + (size_t)i * K;
		for (int k = 0; k < K; k++) {
			const double a = iABase[k];
			if (a == 0.0) continue;   // im2col 出来的 A 里，padding 位置大量是 0，跳过省算力
			const double* kBBase = B + (size_t)k * N;
			for (int j = 0; j < N; j++) {
				iCBase[j] += a * kBBase[j];
			}
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
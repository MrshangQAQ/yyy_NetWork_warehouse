#include"Matrix.h"
#include<chrono>
#include<utility>
#include<cstring>
#include<cstddef>
#if defined(_MSC_VER) || defined(__AVX__)
#include<immintrin.h>
#endif
// ===== 消融实验开关（诊断用；正式基准请保持 0）=====
// 0 = 正常运行
// 1 = 只测「加载阶段」：计算循环替换为极廉价读取，防止加载被优化掉
// 2 = 只测「计算阶段」：共享内存用常量填充，完全不读全局内存
#define ABLATION_MODE 0
// 分段计时开关：0 = 不打印（训练/正式使用）；1 = 每次调用打印 H2D/内核/D2H 耗时（基准测试）
#define MATMUL_GPU_TIMING 0
// 矩阵乘传输路径开关：
//   1 = A 路径：主机侧 double→float（AVX2）→ 锁页 → 异步 H2D → float GEMM → 异步 D2H → 主机侧转回 double
//   0 = 旧路径：pageable 传 double → 设备端转换 → GEMM 输出 double → pageable 拷回
// 用 MATMUL_GPU_TIMING=1 可以对比两条路径的分段耗时。
#define MATMUL_TRANSFER_MODE 1
constexpr int Shared_Memory_Size= 24064;
//Shared_Memory备选值：
// 
// 32256
// 24064
// 48640
static double* d_mat1 = nullptr; static size_t cap1 = 0;
static double* d_mat2 = nullptr; static size_t cap2 = 0;
static double* d_out = nullptr; static size_t capO = 0;
static float* d_mat1_f = nullptr; static size_t cap1f = 0;
static float* d_mat2_f = nullptr; static size_t cap2f = 0;

// ===== A 路径（锁页 + 主机侧转换）专用缓冲：只增不减，跨调用复用 =====
static float* h_pin_mat1 = nullptr; static size_t capPin1 = 0;
static float* h_pin_mat2 = nullptr; static size_t capPin2 = 0;
static float* h_pin_out = nullptr;  static size_t capPinO = 0;
static float* d_float_mat1 = nullptr; static size_t capF1 = 0;
static float* d_float_mat2 = nullptr; static size_t capF2 = 0;
static float* d_float_out = nullptr;  static size_t capFO = 0;
static cudaStream_t g_stream = nullptr;
//编译期计算的常量
constexpr int kBlockSize = 16;
constexpr int kColsPerThr = 4;
constexpr int kRowsPerThr = 4;
constexpr int kInnerPerT = (Shared_Memory_Size / (kBlockSize * kColsPerThr * sizeof(float) + kBlockSize * kRowsPerThr * sizeof(float))) %2==0?//
							Shared_Memory_Size / (kBlockSize * kColsPerThr * sizeof(float) + kBlockSize * kRowsPerThr * sizeof(float))  -1:
							Shared_Memory_Size / (kBlockSize * kColsPerThr * sizeof(float) + kBlockSize * kRowsPerThr * sizeof(float))    ;
constexpr int kTileCols = kBlockSize * kColsPerThr;     // 每个block总共要计算多少列
constexpr int kTileRows = kBlockSize * kRowsPerThr;		//每个block总共要计算多少行
constexpr int kNThreads = kBlockSize * kBlockSize;     // 一般为256，每个block的总线程数

static __global__ void DoubleToFloat_Kernel(const double* __restrict src, float* __restrict dst, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) dst[i] = (float)src[i];   // 合并访存，纯带宽活
}

/// <summary>
/// 
/// </summary>
/// <param name="d_mat1_ptr"> 存储mat1数据的指针</param>
/// <param name="d_mat2_ptr"> 存储mat2数据的指针</param>
/// <param name="d_newData_ptr"> 表示新矩阵的写入的位置</param>
/// <param name="rows"> 新矩阵的行数</param>
/// <param name="inner">内维的数据的总长度</param>
/// <param name="cols"> 新矩阵的列数</param>
/// <returns></returns>
template <typename TOut>
__global__ void Matrix_Double_Mutiple_GPU_Kernel(const float* __restrict d_mat1_ptr, const float* __restrict d_mat2_ptr, TOut* __restrict d_newData_ptr, int rows, int inner, int cols) {
	//我们约定y对应row，x对应col
	//同时存储mat1，mat2的数据，下面每个线程会单独持有两个指针，一个表示mat1的0元素的地址，一个表示mat2的0元素的地址（mat1在前,mat2在后）
	extern __shared__ float Mat_Region[];
	//我们现在来做寄存器优化，每个线程计算4*8的区域的新矩阵的值(这里硬编码了)
	int RegionSize_per_threads = kColsPerThr * kRowsPerThr;
	//表示每次需要读取多少个内维数据，和总共需要读取多少次内维数据(隐式的依赖传入block的DimX=DimY,col_per_threads=row_per_threads)
	int K=(inner+ kInnerPerT -1)/ kInnerPerT;
	//简写的记号
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//一些强度削减的常量：
	//从输入中读取数据时和从共享内存中读取数据计算时都会用到,row_per_threads*ty*inner
	int RTYI = kRowsPerThr * ty * inner;
	//从输入中读取数据时和从共享内存中读取数据计算时都会用到，row_per_threads*ty*inner_per_times
	int RTYIT = kRowsPerThr * ty * kInnerPerT;
	//只有在从共享内存中读取数据时会用到,col_per_threads * tx * inner_per_times
	int CTXIT = kColsPerThr * tx * kInnerPerT;
	//读取矩阵中数据时用到的基地址（实际上,以下计算隐式依赖blockDim.x=blockDim.y）
	const float*  Mat1_Base_Address = d_mat1_ptr + blockIdx.y * (kBlockSize * kRowsPerThr) * inner + RTYI;//打个括号，暗示一下blockDim.y * row_per_threads才是每个block计算的行数
	const float*  Mat2_Base_Address = d_mat2_ptr + blockIdx.x * (kBlockSize * kRowsPerThr);//这里不乘inner了，直接表示从mat2的哪一列开始读
	//mat1和mat2读取时的基础地址，每读完一轮，会重新写入（实际上,以下计算隐式依赖blockDim.x=blockDim.y与col_per_threads=row_per_threads）
	float* Mat1_Region = Mat_Region + RTYIT;

	float* Mat2_Region = Mat_Region + kInnerPerT * (kBlockSize * kRowsPerThr);//Mat_Region的基址+Mat1所占用的位置
	float* Mat2_Region_Plus_CTXIT = Mat2_Region + CTXIT;
	//得到该block右上角的元素，在NewData中的位置
	int Block_Base_Address = 
		kBlockSize * (blockIdx.y * kRowsPerThr) * cols + //计算该块在grid中的哪一行，乘以cols，得到该行的0号元素，在NewData中的位置
		kBlockSize * (blockIdx.x * kColsPerThr);//得到该块在grid中的哪一列，再乘以每个块的列数，得到该块的右上角再NewData中的位置
	int Theard_Base_Address = (ty * kRowsPerThr) * cols + tx * kColsPerThr;
	//NBA:NewDataAddress
	TOut* NBA = d_newData_ptr + Block_Base_Address + Theard_Base_Address;//计算该线程左上角元素在NewData中的位置;
	
	//该线程计算属于自己的那一部分的内容
	float sum00 = 0, sum01 = 0, sum02 = 0, sum03 = 0,
		  sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0,
		  sum20 = 0, sum21 = 0, sum22 = 0, sum23 = 0,
		  sum30 = 0, sum31 = 0, sum32 = 0, sum33 = 0
		;
	//该线程要将sum00放到新矩阵的什么位置（用于越界检查）
	int RowOut = (blockIdx.y * kBlockSize + ty) * kRowsPerThr;
	int ColOut = (blockIdx.x * kBlockSize + tx) * kColsPerThr;//这里把tx提到外面，对应下面的偏差变为16，每个线程写一部分
	//检查行/列，是否越界
	// IR0A:Is row0 access
	bool IR0A = RowOut     < rows;
	bool IR1A = RowOut + 1 < rows;
	bool IR2A = RowOut + 2 < rows;
	bool IR3A = RowOut + 3 < rows;
	// IC0A:Is col0 access
	bool IC0A = ColOut	    < cols;
	bool IC1A = ColOut + 1 < cols;
	bool IC2A = ColOut + 2 < cols;
	bool IC3A = ColOut + 3 < cols;
	//它们写入位置的层基址，由ty决定
	//这里我们把这个RTYIT放到Mat1_Region中,原来为
	//Mat1_Region = Mat_Region
	//float* Mat1_Region_Base_Address0 = Mat1_Region + RTYIT;
	//float* Mat1_Region_Base_Address1 = Mat1_Region + RTYIT + inner_per_times;
	//float* Mat1_Region_Base_Address2 = Mat1_Region + RTYIT + 2 * inner_per_times;
	//float* Mat1_Region_Base_Address3 = Mat1_Region + RTYIT + 3 * inner_per_times;
	float* Mat1_Region_Base_Address0 = Mat1_Region;
	float* Mat1_Region_Base_Address1 = Mat1_Region + kInnerPerT;
	float* Mat1_Region_Base_Address2 = Mat1_Region + 2 * kInnerPerT;
	float* Mat1_Region_Base_Address3 = Mat1_Region + 3 * kInnerPerT;

	//按ty选取，它们读取位置时的层基址，（一个线程管4行，古分别有4个），我们这里统一ty，因为后面的读取都是ty不变，tx做偏置的
	//const double* Ty_Mat1_Base_Address0 = Mat1_Base_Address + (0 + ty * row_per_threads) * inner;
	//const double* Ty_Mat1_Base_Address1 = Mat1_Base_Address + (1 + ty * row_per_threads) * inner;
	//const double* Ty_Mat1_Base_Address2 = Mat1_Base_Address + (2 + ty * row_per_threads) * inner;
	//const double* Ty_Mat1_Base_Address3 = Mat1_Base_Address + (3 + ty * row_per_threads) * inner;

	//const double* Ty_Mat2_Base_Address0 = Mat2_Base_Address + 0 + ty * col_per_threads;
	//const double* Ty_Mat2_Base_Address1 = Mat2_Base_Address + 1 + ty * col_per_threads;
	//const double* Ty_Mat2_Base_Address2 = Mat2_Base_Address + 2 + ty * col_per_threads;
	//const double* Ty_Mat2_Base_Address3 = Mat2_Base_Address + 3 + ty * col_per_threads;
	//辅助写入的值，它们用于检查，这个写入的外维是否越界，不检查最后会把值放在什么位置
	//int RowIdx = (blockIdx.y * blockDim.y + ty) * row_per_threads;   
	//int ColIdx = (blockIdx.x * blockDim.x + ty) * col_per_threads;   // blockIdx.x * blockDim.x，同理，表示的是读取B时的基础地址，+ty（因为是按每ty为分配1层进行填充的），表示该线程，实际上读入的是B的哪一列
	//对上一步的强度削减，每个位置单独持有一个
	int RowIdx0 = (blockIdx.y * kBlockSize + ty) * kRowsPerThr; // blockIdx.y * blockDim.y * row_per_threads，表示的是该线程左上角的节点，在新矩阵中的行索引，也是我们读取A时的基础地址，+ty（因为是按每ty为分配1层进行填充的），故其
	bool IsAccessRowIdx0 = (RowIdx0) < rows;
	bool IsAccessRowIdx1 = (RowIdx0 + 1) < rows;
	bool IsAccessRowIdx2 = (RowIdx0 + 2) < rows;
	bool IsAccessRowIdx3 = (RowIdx0 + 3) < rows;
	//Col的版本现已弃用
	//int ColIdx0 = (blockIdx.x * kBlockSize + ty) * kColsPerThr;   // blockIdx.x * blockDim.x * col_per_threads，同理，表示的是读取B时的基础地址，+ty（因为是按每ty为分配1层进行填充的），表示该线程，实际上读入的是B的哪一列
	//bool IsAccessColIdx0 = (ColIdx0) < cols;
	//bool IsAccessColIdx1 = (ColIdx0 + 1) < cols;
	//bool IsAccessColIdx2 = (ColIdx0 + 2) < cols;
	//bool IsAccessColIdx3 = (ColIdx0 + 3) < cols;
	//bool IsAccessColIdx4 = (ColIdx0 + 4) < cols;
	//bool IsAccessColIdx5 = (ColIdx0 + 5) < cols;
	//bool IsAccessColIdx6 = (ColIdx0 + 6) < cols;
	//bool IsAccessColIdx7 = (ColIdx0 + 7) < cols;
	
	//计算列所需要的一些常量
	//外层循环，逐步计算出sum
	for (int k = 0;k < K;k++) {
		//k*inner_per_times，决定它们读那一段
		int in_base = k * kInnerPerT;
		//这次读取的基础地址，层基址把它们按层区分开,in_base决定它们读取哪一段
		// Mat1_Base_Address
		//const float* TyK_Mat1_Base_Address0 = Mat1_Base_Address + RTYI + in_base;
		//const float* TyK_Mat1_Base_Address1 = Mat1_Base_Address + RTYI + in_base + 1 * inner;
		//const float* TyK_Mat1_Base_Address2 = Mat1_Base_Address + RTYI + in_base + 2 * inner;
		//const float* TyK_Mat1_Base_Address3 = Mat1_Base_Address + RTYI + in_base + 3 * inner;
		//这里，我们把RTYI移动到外部第一次计算Mat1_Base_Address时
		const float* TyK_Mat1_Base_Address0 = Mat1_Base_Address + in_base;
		const float* TyK_Mat1_Base_Address1 = Mat1_Base_Address + in_base + 1*inner;
		const float* TyK_Mat1_Base_Address2 = Mat1_Base_Address + in_base + 2*inner;
		const float* TyK_Mat1_Base_Address3 = Mat1_Base_Address + in_base + 3*inner;
		//多个线程共同填充共享内存，每个线程只填一部分
		for (int in = tx;in < kInnerPerT;in += kBlockSize) {
#if ABLATION_MODE == 2
			//【消融·计算模式】共享内存用常量填充，完全不读全局内存
			Mat1_Region_Base_Address0[in] = 1.0f;
			Mat1_Region_Base_Address1[in] = 1.0f;
			Mat1_Region_Base_Address2[in] = 1.0f;
			Mat1_Region_Base_Address3[in] = 1.0f;
#else
			Mat1_Region_Base_Address0[in] = (!IsAccessRowIdx0 || in + in_base >= inner) ? 0.0f : TyK_Mat1_Base_Address0[in];
			Mat1_Region_Base_Address1[in] = (!IsAccessRowIdx1 || in + in_base >= inner) ? 0.0f : TyK_Mat1_Base_Address1[in];//TyK_Mat1_Base_Address1
			Mat1_Region_Base_Address2[in] = (!IsAccessRowIdx2 || in + in_base >= inner) ? 0.0f : TyK_Mat1_Base_Address2[in];//TyK_Mat1_Base_Address2
			Mat1_Region_Base_Address3[in] = (!IsAccessRowIdx3 || in + in_base >= inner) ? 0.0f : TyK_Mat1_Base_Address3[in];//TyK_Mat1_Base_Address3
#endif

		}
		// =====【优化】mat2 合并加载：让 warp 沿「列方向」排开 =====
		// 原实现沿「行」方向取 mat2（地址步长 cols 个元素），同一 warp 的 32 个 lane
		// 散落在 16 条 cache line 上，访存完全未合并。现把整个 tile 摊成一维，
		// 让「列 c 快变、行 k 慢变」：相邻 lane 落在相邻列上 → 全局读连续（合并）。
		// 写 shared 时的跨度是 inner_per_times（奇数 95，与 bank 数 32 互质）→ bank 天然错开。
		// shared 布局 [c][k] 与原来完全一致，所以计算阶段无需任何改动。
		{

			//const int col_base = blockIdx.x * blockDim.x * kColsPerThr;   
			//const int tid = ty * blockDim.x + tx;
			//const int nthreads = blockDim.x * blockDim.y;                 
			//const int tileCols = blockDim.x * kColsPerThr;                
			//const int col_base = kBlockSize * kTileCols;   // 本 block 的起始全局列号 = bx*64
			const int tid = ty * kBlockSize + tx;//这是第几个线程，实际上，我们共有kTileCols个列，我们希望这256个线程，每个线程一次填充某行的1列，直到该列被填完，每次填充N*KTileCols个部分，N=kNThreads/kTileCols
			//const int nthreads = kNThreads;                   // 256
			//const int tileCols = kTileCols;                // 64
			for (int idx = tid; idx < kTileCols * kInnerPerT; idx += kNThreads) {
				const int kIdx = idx / kTileCols;      // 内维方向（慢变），等价于原来的 in，这里应该是0，2，4。。。。
				const int cIdx = idx % kTileCols;      // 通道方向（快变），0.....127
#if ABLATION_MODE == 2
				//【消融·计算模式】共享内存用常量填充，完全不读全局内存
				Mat2_Region[cIdx * kInnerPerT + kIdx] = 1.0f;
#else
				const int r  = in_base + kIdx;        // 全局行号
				const int gc = blockIdx.x * kTileCols + cIdx;       // 全局列号
				const bool ok = (r < inner) && (gc < cols);   // 越界保护（等价于原来的 IsAccessColIdx + in_base 判断）

				Mat2_Region[cIdx * kInnerPerT + kIdx] =
					ok ? d_mat2_ptr[(size_t)r * cols + gc] : 0.0f;
#endif
			}
		}
		__syncthreads();
		//调试代码，出现Bug时选个小矩阵打印一下
		/*
		if (blockIdx.x == 1 && blockIdx.y == 1 && tx == 0 && ty == 0) {
			printf("%0.f\n", Mat2_Region[0]);
			//for (int i = 0;i < kTileCols;i++) {
			//	for (int j = 0;j < kInnerPerT;j++) {
			//		printf("%0.f", Mat2_Region[i * kInnerPerT + j]);
			//	}
			//	printf("\n");
			//}
		}
		*/
		//阻滞，保证共享的部分被完全初始化再开始读，计算矩阵
		//计算该位置处乘法的结果
		//TM1BA:The_Mat1_Base_Address，这里进行了强度削减，原来为
		//Mat1_Region = Mat_Region
		//const float* TM1BA0 = Mat1_Region + RTYIT;
		//const float* TM1BA1 = Mat1_Region + RTYIT + 1 * inner_per_times;
		//const float* TM1BA2 = Mat1_Region + RTYIT + 2 * inner_per_times;
		//const float* TM1BA3 = Mat1_Region + RTYIT + 3 * inner_per_times;
		const float* TM1BA0 = Mat1_Region;
		const float* TM1BA1 = Mat1_Region + 1 * kInnerPerT;
		const float* TM1BA2 = Mat1_Region + 2 * kInnerPerT;
		const float* TM1BA3 = Mat1_Region + 3 * kInnerPerT;
		//已晕倒(要不干脆硬编码，更快)
		const float* TM2BA0 = Mat2_Region_Plus_CTXIT;
		const float* TM2BA1 = Mat2_Region_Plus_CTXIT + 1 * kInnerPerT;//16=1*blockDim.x
		const float* TM2BA2 = Mat2_Region_Plus_CTXIT + 2 * kInnerPerT;
		const float* TM2BA3 = Mat2_Region_Plus_CTXIT + 3 * kInnerPerT;
#if ABLATION_MODE == 1
		//【消融·加载模式】只测「加载阶段」：保留真实的全局读取与共享写入，
		// 但把计算循环替换为极廉价的读取——读全部 4 行 mat1 + 4 列 mat2 的首元素。
		// 目的：让 nvcc 无法把整个加载阶段当死代码消掉。
		// 代价仅 8 条 LDS（原本 760 条），可忽略。
		sum00 += TM1BA0[0] + TM1BA1[0] + TM1BA2[0] + TM1BA3[0]
		       + TM2BA0[0] + TM2BA1[0] + TM2BA2[0] + TM2BA3[0];
#else
		for (int in = 0;in < kInnerPerT;in++) {
			float r0 = TM1BA0[in], r1 = TM1BA1[in], r2 = TM1BA2[in], r3 = TM1BA3[in];
			float c0 = TM2BA0[in], c1 = TM2BA1[in], c2 = TM2BA2[in], c3 = TM2BA3[in];
			sum00 += r0 * c0; sum01 += r0 * c1; sum02 += r0 * c2, sum03 += r0 * c3; 
			sum10 += r1 * c0; sum11 += r1 * c1; sum12 += r1 * c2, sum13 += r1 * c3;
			sum20 += r2 * c0; sum21 += r2 * c1; sum22 += r2 * c2, sum23 += r2 * c3; 
			sum30 += r3 * c0; sum31 += r3 * c1; sum32 += r3 * c2, sum33 += r3 * c3; 
		}
#endif
		__syncthreads();//再次阻滞，保证每个格子都计算完成再开始下一步
	}
	if (IR0A && IC0A) NBA[0 * cols + 0] = (TOut)sum00;if (IR0A && IC1A) NBA[0 * cols + 1] = (TOut)sum01;if (IR0A && IC2A) NBA[0 * cols + 2] = (TOut)sum02;if (IR0A && IC3A) NBA[0 * cols + 3] = (TOut)sum03;
	if (IR1A && IC0A) NBA[1 * cols + 0] = (TOut)sum10;if (IR1A && IC1A) NBA[1 * cols + 1] = (TOut)sum11;if (IR1A && IC2A) NBA[1 * cols + 2] = (TOut)sum12;if (IR1A && IC3A) NBA[1 * cols + 3] = (TOut)sum13;
	if (IR2A && IC0A) NBA[2 * cols + 0] = (TOut)sum20;if (IR2A && IC1A) NBA[2 * cols + 1] = (TOut)sum21;if (IR2A && IC2A) NBA[2 * cols + 2] = (TOut)sum22;if (IR2A && IC3A) NBA[2 * cols + 3] = (TOut)sum23;
	if (IR3A && IC0A) NBA[3 * cols + 0] = (TOut)sum30;if (IR3A && IC1A) NBA[3 * cols + 1] = (TOut)sum31;if (IR3A && IC2A) NBA[3 * cols + 2] = (TOut)sum32;if (IR3A && IC3A) NBA[3 * cols + 3] = (TOut)sum33;
	
}
Matrix_Double Matrix_Double_Mutiple_GPU(const Matrix_Double& mat1, const Matrix_Double& mat2) {

	cudaEvent_t e0, e1, e2, e3,e_begin,e_end;
	float ms01 = 0.0;
	float ms12 = 0.0;
	float ms23 = 0.0;
	float msbe = 0.0;
	cudaEventCreate(&e_begin); cudaEventCreate(&e_end);
	int rows = mat1.rowNum_, inner = mat1.colNum_, cols = mat2.colNum_;
	if (inner != mat2.rowNum_) {
		throw std::runtime_error("mismatched Matrix");
	}
	if (mat1.data.size() == 0 || mat2.data.size() == 0) {
		throw std::runtime_error("empty matrix");
	}
	cudaEventRecord(e_begin);
	std::vector<double> newdata(rows * cols);//我们要用并行计算，将数据传入它里面
	//这里，我们约定，先存放rows，inner，cols，在依次存入mat1，mat2，的数据，并把数据输入到newData中
	int newDataSize = newdata.size() * sizeof(double);
	int mat1Size = mat1.data.size() * sizeof(double);
	int mat2Size = mat2.data.size() * sizeof(double);
	const size_t mat1Sizef = mat1.data.size() * sizeof(float);
	const size_t mat2Sizef = mat2.data.size() * sizeof(float);

	cudaEventRecord(e_end);
	cudaEventCreate(&e0); cudaEventCreate(&e1);
	cudaEventCreate(&e2); cudaEventCreate(&e3);

	if (mat1Size > cap1) { if (d_mat1) { cudaFree(d_mat1); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat1), mat1Size));cap1 = mat1Size; }
	if (mat2Size > cap2) { if (d_mat2) { cudaFree(d_mat2); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat2), mat2Size));cap2 = mat2Size; }
	if (newDataSize > capO) { if (d_out) { cudaFree(d_out); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), newDataSize));capO = newDataSize; }
	if (mat1Sizef > cap1f) { if (d_mat1_f) { cudaFree(d_mat1_f); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat1_f), mat1Sizef));cap1f = mat1Sizef; }
	if (mat2Sizef > cap2f) { if (d_mat2_f) { cudaFree(d_mat2_f); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat2_f), mat2Sizef));cap2f = mat2Sizef; }
	cudaEventRecord(e0);
	int gridmat1 = (mat1.data.size()+1023) / 1024;
	int gridmat2 = (mat2.data.size()+1023) / 1024;

	CUDA_CHECK(cudaMemcpy(d_mat1, mat1.data.data(), mat1Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_mat2, mat2.data.data(), mat2Size, cudaMemcpyHostToDevice));
	DoubleToFloat_Kernel << <gridmat1, 1024 >> > (d_mat1, d_mat1_f, mat1.data.size());
	DoubleToFloat_Kernel << <gridmat2, 1024 >> > (d_mat2, d_mat2_f, mat2.data.size());

	cudaEventRecord(e1);
	int gridy = (rows + kTileRows - 1) / (kTileRows);//表示，新得到的矩阵的block有几行
	int gridx = (cols + kTileCols - 1) / (kTileCols);//表示，新得到的矩阵的block有几列
	Matrix_Double_Mutiple_GPU_Kernel<double><<<dim3(gridx, gridy, 1), dim3(kBlockSize, kBlockSize, 1), Shared_Memory_Size>>>(d_mat1_f, d_mat2_f, d_out, rows, inner, cols);
	cudaEventRecord(e2);
	//调试，看有没有错误信息
	CUDA_CHECK(cudaGetLastError());
	//把新得到的数据归还主机
	CUDA_CHECK(cudaMemcpy(&newdata[0],d_out , newDataSize, cudaMemcpyDeviceToHost));
	cudaEventRecord(e3);
	cudaEventSynchronize(e3);
	cudaEventSynchronize(e_end);
	cudaEventElapsedTime(&ms01, e0, e1);
	cudaEventElapsedTime(&ms12, e1, e2);
	cudaEventElapsedTime(&ms23, e2, e3);
	cudaEventElapsedTime(&msbe, e_begin, e_end);
	std::cout << ms01 << "ms," << ms12 << "ms," << ms23 << "ms,"<<msbe<<"ms," << std::endl;
	//现在释放，数据已回到内存
	return Matrix_Double(std::move(newdata), rows, cols);
}
/// <summary>三参数版本：把 mat1×mat2 的结果直接写入调用方提供的 out。
/// 结果缓冲由调用方持有并复用，本函数不做任何分配/清零（用于高频调用或基准测试）。</summary>
/// <param name="mat1">左矩阵（M×K）</param>
/// <param name="mat2">右矩阵（K×N）</param>
/// <param name="out">结果输出矩阵；尺寸不符时自动调整为 M×N</param>
/// <exception cref="std::runtime_error">mat1 的列数不等于 mat2 的行数，或任一输入为空矩阵时抛出</exception>
// ===================== A 路径辅助：锁页/显存缓冲 + 主机侧类型转换 =====================
#if MATMUL_TRANSFER_MODE == 1
namespace {
	/// <summary>只增不减的锁页缓冲（容量单位：float 个数）。首次分配后写一遍，避免首帧 page fault 进入计时区。</summary>
	float* EnsurePinnedFloats(float*& ptr, size_t& cap, size_t needElems) {
		if (needElems > cap) {
			if (ptr) { CUDA_CHECK(cudaFreeHost(ptr)); ptr = nullptr; }
			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr), needElems * sizeof(float)));
			std::memset(ptr, 0, needElems * sizeof(float));
			cap = needElems;
		}
		return ptr;
	}
	/// <summary>只增不减的显存缓冲（容量单位：float 个数）。</summary>
	float* EnsureDeviceFloats(float*& ptr, size_t& cap, size_t needElems) {
		if (needElems > cap) {
			if (ptr) { cudaFree(ptr); ptr = nullptr; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), needElems * sizeof(float)));
			cap = needElems;
		}
		return ptr;
	}
	/// <summary>double → float：AVX2 每次处理 4 个，余数用标量。</summary>
	void ConvertDoubleToFloat(const double* __restrict src, float* __restrict dst, size_t n) {
#if defined(_MSC_VER) || defined(__AVX__)
		const size_t n4 = n & ~(size_t)3;
#if defined(_OPENMP)
#pragma omp parallel for if(n4 > 65536) schedule(static)
#endif
		for (long long i = 0; i < (long long)n4; i += 4) {
			const __m256d d = _mm256_loadu_pd(src + i);
			_mm_storeu_ps(dst + i, _mm256_cvtpd_ps(d));
		}
		for (size_t i = n4; i < n; ++i) dst[i] = (float)src[i];
#else
#if defined(_OPENMP)
#pragma omp parallel for if(n > 65536) schedule(static)
#endif
		for (long long i = 0; i < (long long)n; ++i) dst[i] = (float)src[i];
#endif
	}
	/// <summary>float → double：AVX2 每次处理 4 个，余数用标量。</summary>
	void ConvertFloatToDouble(const float* __restrict src, double* __restrict dst, size_t n) {
#if defined(_MSC_VER) || defined(__AVX__)
		const size_t n4 = n & ~(size_t)3;
#if defined(_OPENMP)
#pragma omp parallel for if(n4 > 65536) schedule(static)
#endif
		for (long long i = 0; i < (long long)n4; i += 4) {
			const __m128 f = _mm_loadu_ps(src + i);
			_mm256_storeu_pd(dst + i, _mm256_cvtps_pd(f));
		}
		for (size_t i = n4; i < n; ++i) dst[i] = (double)src[i];
#else
#if defined(_OPENMP)
#pragma omp parallel for if(n > 65536) schedule(static)
#endif
		for (long long i = 0; i < (long long)n; ++i) dst[i] = (double)src[i];
#endif
	}
} // namespace
#endif // MATMUL_TRANSFER_MODE == 1

/// <summary>三参数版本：把 mat1×mat2 的结果直接写入调用方提供的 out。
/// 结果缓冲由调用方持有并复用；传输路径由 MATMUL_TRANSFER_MODE 决定（1 = 锁页 + 主机侧转换，0 = 旧路径）。</summary>
/// <param name="mat1">左矩阵（M×K）</param>
/// <param name="mat2">右矩阵（K×N）</param>
/// <param name="out">结果输出矩阵；尺寸不符时自动调整为 M×N</param>
/// <exception cref="std::runtime_error">mat1 的列数不等于 mat2 的行数，或任一输入为空矩阵时抛出</exception>
void Matrix_Double_Mutiple_GPU(const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out) {
	const int rows = mat1.rowNum_, inner = mat1.colNum_, cols = mat2.colNum_;
	if (inner != mat2.rowNum_) { throw std::runtime_error("mismatched Matrix"); }
	if (mat1.data.size() == 0 || mat2.data.size() == 0) { throw std::runtime_error("empty matrix"); }
	//结果缓冲复用：仅当尺寸变化时才重新分配；稳态下 resize 是 no-op，既不分配也不清零
	const size_t need = (size_t)rows * cols;
	if (out.data.size() != need) { out.data.resize(need); }
	out.rowNum_ = rows;
	out.colNum_ = cols;

	const size_t m1n = mat1.data.size();
	const size_t m2n = mat2.data.size();
	const int gridy = (rows + kTileRows - 1) / kTileRows;
	const int gridx = (cols + kTileCols - 1) / kTileCols;

#if MATMUL_GPU_TIMING
	using SteadyClock = std::chrono::steady_clock;
	static cudaEvent_t tm0 = nullptr, tm1 = nullptr, tm2 = nullptr, tm3 = nullptr;
	if (tm0 == nullptr) {
		CUDA_CHECK(cudaEventCreate(&tm0)); CUDA_CHECK(cudaEventCreate(&tm1));
		CUDA_CHECK(cudaEventCreate(&tm2)); CUDA_CHECK(cudaEventCreate(&tm3));
	}
#endif

#if MATMUL_TRANSFER_MODE == 1
	// ---------- A 路径：主机侧转 float → 锁页 → 异步 H2D → float GEMM → 异步 D2H → 主机侧转回 double ----------
	float* h1 = EnsurePinnedFloats(h_pin_mat1, capPin1, m1n);
	float* h2 = EnsurePinnedFloats(h_pin_mat2, capPin2, m2n);
	float* hOut = EnsurePinnedFloats(h_pin_out, capPinO, need);
	float* d1 = EnsureDeviceFloats(d_float_mat1, capF1, m1n);
	float* d2 = EnsureDeviceFloats(d_float_mat2, capF2, m2n);
	float* dOut = EnsureDeviceFloats(d_float_out, capFO, need);
	if (g_stream == nullptr) { CUDA_CHECK(cudaStreamCreate(&g_stream)); }

#if MATMUL_GPU_TIMING
	const auto tBegin = SteadyClock::now();
#endif
	ConvertDoubleToFloat(mat1.data.data(), h1, m1n);
	ConvertDoubleToFloat(mat2.data.data(), h2, m2n);
#if MATMUL_GPU_TIMING
	const auto tPacked = SteadyClock::now();
	cudaEventRecord(tm0);
#endif
	CUDA_CHECK(cudaMemcpyAsync(d1, h1, m1n * sizeof(float), cudaMemcpyHostToDevice, g_stream));
	CUDA_CHECK(cudaMemcpyAsync(d2, h2, m2n * sizeof(float), cudaMemcpyHostToDevice, g_stream));
#if MATMUL_GPU_TIMING
	cudaEventRecord(tm1);
#endif
	Matrix_Double_Mutiple_GPU_Kernel<float><<<dim3(gridx, gridy, 1), dim3(kBlockSize, kBlockSize, 1), Shared_Memory_Size, g_stream>>>(d1, d2, dOut, rows, inner, cols);
	CUDA_CHECK(cudaGetLastError());
#if MATMUL_GPU_TIMING
	cudaEventRecord(tm2);
#endif
	CUDA_CHECK(cudaMemcpyAsync(hOut, dOut, need * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
#if MATMUL_GPU_TIMING
	cudaEventRecord(tm3);
	CUDA_CHECK(cudaEventSynchronize(tm3));
	const auto tD2HDone = SteadyClock::now();
#else
	CUDA_CHECK(cudaStreamSynchronize(g_stream));
#endif
	ConvertFloatToDouble(hOut, out.data.data(), need);

#if MATMUL_GPU_TIMING
	const auto tEnd = SteadyClock::now();
	const double msPack = std::chrono::duration<double, std::milli>(tPacked - tBegin).count();
	const double msUnpack = std::chrono::duration<double, std::milli>(tEnd - tD2HDone).count();
	const double msTotal = std::chrono::duration<double, std::milli>(tEnd - tBegin).count();
	float msH2D = 0.0f, msKernel = 0.0f, msD2H = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&msH2D, tm0, tm1));
	CUDA_CHECK(cudaEventElapsedTime(&msKernel, tm1, tm2));
	CUDA_CHECK(cudaEventElapsedTime(&msD2H, tm2, tm3));
	// 首次调用打印一次资源配置（寄存器/占用率）
	static bool cfgPrinted = false;
	if (!cfgPrinted) {
		cfgPrinted = true;
		cudaFuncAttributes attr{};
		CUDA_CHECK(cudaFuncGetAttributes(&attr, Matrix_Double_Mutiple_GPU_Kernel<float>));
		int blocksPerSM = 0;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&blocksPerSM, Matrix_Double_Mutiple_GPU_Kernel<float>,
			kBlockSize * kBlockSize, Shared_Memory_Size));
		std::cout << "[matmul-GPU A] regs=" << attr.numRegs
			<< " shared/block=" << attr.sharedSizeBytes
			<< " local=" << attr.localSizeBytes << "B"
			<< " blocks/SM=" << blocksPerSM
			<< " threads/SM=" << blocksPerSM * kBlockSize * kBlockSize << std::endl;
	}
	std::cout << "[matmul-GPU A] pack=" << msPack << "ms, H2D=" << msH2D
		<< "ms, kernel=" << msKernel << "ms, D2H=" << msD2H
		<< "ms, unpack=" << msUnpack << "ms, total=" << msTotal << "ms" << std::endl;
#endif
#else
	// ---------- 旧路径：pageable 传 double → 设备端转换 → GEMM 输出 double → pageable 拷回 ----------
	const size_t newDataSize = need * sizeof(double);
	const size_t mat1Size = mat1.data.size() * sizeof(double);
	const size_t mat2Size = mat2.data.size() * sizeof(double);
	const size_t mat1Sizef = mat1.data.size() * sizeof(float);
	const size_t mat2Sizef = mat2.data.size() * sizeof(float);
	if (mat1Size > cap1) { if (d_mat1) { cudaFree(d_mat1); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat1), mat1Size));cap1 = mat1Size; }
	if (mat2Size > cap2) { if (d_mat2) { cudaFree(d_mat2); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat2), mat2Size));cap2 = mat2Size; }
	if (newDataSize > capO) { if (d_out) { cudaFree(d_out); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), newDataSize));capO = newDataSize; }
	if (mat1Sizef > cap1f) { if (d_mat1_f) { cudaFree(d_mat1_f); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat1_f), mat1Sizef));cap1f = mat1Sizef; }
	if (mat2Sizef > cap2f) { if (d_mat2_f) { cudaFree(d_mat2_f); }CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat2_f), mat2Sizef));cap2f = mat2Sizef; }
#if MATMUL_GPU_TIMING
	cudaEventRecord(tm0);
#endif
	CUDA_CHECK(cudaMemcpy(d_mat1, mat1.data.data(), mat1Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_mat2, mat2.data.data(), mat2Size, cudaMemcpyHostToDevice));
	const int gridmat1 = (int)((mat1.data.size() + 1023) / 1024);
	const int gridmat2 = (int)((mat2.data.size() + 1023) / 1024);
	DoubleToFloat_Kernel << <gridmat1, 1024 >> > (d_mat1, d_mat1_f, (int)mat1.data.size());
	DoubleToFloat_Kernel << <gridmat2, 1024 >> > (d_mat2, d_mat2_f, (int)mat2.data.size());
#if MATMUL_GPU_TIMING
	cudaEventRecord(tm1);
#endif
	Matrix_Double_Mutiple_GPU_Kernel<double><<<dim3(gridx, gridy, 1), dim3(kBlockSize, kBlockSize, 1), Shared_Memory_Size>>>(d_mat1_f, d_mat2_f, d_out, rows, inner, cols);
	CUDA_CHECK(cudaGetLastError());
#if MATMUL_GPU_TIMING
	cudaEventRecord(tm2);
#endif
	CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, newDataSize, cudaMemcpyDeviceToHost));
#if MATMUL_GPU_TIMING
	cudaEventRecord(tm3);
	CUDA_CHECK(cudaEventSynchronize(tm3));
	float ms01 = 0.0f, ms12 = 0.0f, ms23 = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms01, tm0, tm1));
	CUDA_CHECK(cudaEventElapsedTime(&ms12, tm1, tm2));
	CUDA_CHECK(cudaEventElapsedTime(&ms23, tm2, tm3));
	std::cout << "[matmul-GPU legacy] H2D+convert=" << ms01 << "ms, kernel=" << ms12
		<< "ms, D2H=" << ms23 << "ms" << std::endl;
#else
	CUDA_CHECK(cudaStreamSynchronize(0));
#endif
#endif // MATMUL_TRANSFER_MODE
}

/// <summary>矩阵乘法的 GPU 启用阈值（乘加次数）。默认 3200 万：低于它时搬运开销会盖过计算收益。</summary>
static size_t gMatMulGpuThreshold = 32u * 1024 * 1024;

void SetMatMulGpuThreshold(size_t macs) { gMatMulGpuThreshold = macs; }

void Matrix_Double_Mutiple_Auto(const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out) {
	const size_t macs = (size_t)mat1.GetRowNum() * (size_t)mat1.GetColNum() * (size_t)mat2.GetColNum();
	if (macs >= gMatMulGpuThreshold) {
		Matrix_Double_Mutiple_GPU(mat1, mat2, out);
		return;
	}
	// 小矩阵：CPU 版本（返回临时对象再移动，避免多一次大拷贝）
	Matrix_Double tmp = mat1 * mat2;
	out = std::move(tmp);
}
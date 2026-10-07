#include"Matrix.h"
#include<chrono>
// ===== 消融实验开关（诊断用；正式基准请保持 0）=====
// 0 = 正常运行
// 1 = 只测「加载阶段」：计算循环替换为极廉价读取，防止加载被优化掉
// 2 = 只测「计算阶段」：共享内存用常量填充，完全不读全局内存
#define ABLATION_MODE 0
constexpr int Shared_Memory_Size= 24064;
static double* d_mat1 = nullptr; static size_t cap1 = 0;
static double* d_mat2 = nullptr; static size_t cap2 = 0;
static double* d_out = nullptr; static size_t capO = 0;
static float* d_mat1_f = nullptr; static size_t cap1f = 0;
static float* d_mat2_f = nullptr; static size_t cap2f = 0;
//编译期计算的常量
constexpr int kBlockSize = 16;
constexpr int kColsPerThr = 4;
constexpr int kRowsPerThr = 4;
constexpr int kInnerPerT = (Shared_Memory_Size/(2*kBlockSize*kColsPerThr*sizeof(float)))%2==0?// = 49152/(2*4*16*4)→96 取奇 95
			 (Shared_Memory_Size / (2 * kBlockSize * kColsPerThr * sizeof(float)))-1: 
		   	 (Shared_Memory_Size / (2 * kBlockSize * kColsPerThr * sizeof(float)));                          
constexpr int kTileCols = kBlockSize * kColsPerThr;    // 64，每个block总共要计算多少列
constexpr int kNThreads = kBlockSize * kBlockSize;     // 256，每个block的总线程数
__global__ void DoubleToFloat_Kernel(const double* __restrict src, float* __restrict dst, int n) {
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
__global__ void Matrix_Double_Mutiple_GPU_Kernel(const float* __restrict d_mat1_ptr, const float* __restrict d_mat2_ptr, double* d_newData_ptr, int rows, int inner, int cols) {
	//我们约定y对应row，x对应col
	//同时存储mat1，mat2的数据，下面每个线程会单独持有两个指针，一个表示mat1的0元素的地址，一个表示mat2的0元素的地址（mat1在前,mat2在后）
	extern __shared__ float Mat_Region[];
	//我们现在来做寄存器优化，每个线程计算4*4的区域的新矩阵的值(这里硬编码了)
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
	float* Mat2_Region = Mat_Region + kInnerPerT * (kBlockSize * kColsPerThr);
	//得到该block右上角的元素，在NewData中的位置
	int Block_Base_Address = 
		kBlockSize * (blockIdx.y * kColsPerThr) * cols + //计算该块在grid中的哪一行，乘以cols，得到该行的0号元素，在NewData中的位置
		kBlockSize * (blockIdx.x * kRowsPerThr);//得到该块在grid中的哪一列，再乘以每个块的列数，得到该块的右上角再NewData中的位置
	//后面只+tx，写入的时候偏差16，防止撞bank
	int Theard_Base_Address = ty * kColsPerThr * cols + tx * kRowsPerThr;
	//NBA:NewDataAddress
	double* NBA = d_newData_ptr + Block_Base_Address + Theard_Base_Address;//计算该线程左上角元素在NewData中的位置;
	
	//该线程计算属于自己的那一部分的内容
	float sum00 = 0, sum01 = 0, sum02 = 0, sum03 = 0,
		  sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0,
		  sum20 = 0, sum21 = 0, sum22 = 0, sum23 = 0,
		  sum30 = 0, sum31 = 0, sum32 = 0, sum33 = 0
		;
	//该线程要将sum00放到新矩阵的什么位置（用于越界检查）
	int RowOut = (blockIdx.y * kBlockSize + ty) * kColsPerThr;
	int ColOut = (blockIdx.x * kBlockSize + tx) * kRowsPerThr;//这里把tx提到外面，对应下面的偏差变为16，每个线程写一部分
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
	//int RowIdx1 = RowIdx0 + 1;
	//int RowIdx2 = RowIdx0 + 2;
	//int RowIdx3 = RowIdx0 + 3;
	int ColIdx0 = (blockIdx.x * kBlockSize + ty) * kColsPerThr;   // blockIdx.x * blockDim.x * col_per_threads，同理，表示的是读取B时的基础地址，+ty（因为是按每ty为分配1层进行填充的），表示该线程，实际上读入的是B的哪一列
	bool IsAccessColIdx0 = (ColIdx0) < cols;
	bool IsAccessColIdx1 = (ColIdx0 + 1) < cols;
	bool IsAccessColIdx2 = (ColIdx0 + 2) < cols;
	bool IsAccessColIdx3 = (ColIdx0 + 3) < cols;

	//int ColIdx1 = ColIdx0 + 1;
	//int ColIdx2 = ColIdx0 + 2;
	//int ColIdx3 = ColIdx0 + 3;
	
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
			const int tid = ty * kBlockSize + tx;
			//const int nthreads = kNThreads;                   // 256
			//const int tileCols = kTileCols;                // 64
			for (int idx = tid; idx < kTileCols * kInnerPerT; idx += kNThreads) {
				const int kIdx = idx / kTileCols;      // 内维方向（慢变），等价于原来的 in
				const int cIdx = idx % kTileCols;      // 通道方向（快变），0..63
#if ABLATION_MODE == 2
				//【消融·计算模式】共享内存用常量填充，完全不读全局内存
				Mat2_Region[cIdx * kInnerPerT + kIdx] = 1.0f;
#else
				const int r  = in_base + kIdx;        // 全局行号
				const int gc = kBlockSize * kTileCols + cIdx;       // 全局列号
				const bool ok = (r < inner) && (gc < cols);   // 越界保护（等价于原来的 IsAccessColIdx + in_base 判断）

				Mat2_Region[cIdx * kInnerPerT + kIdx] =
					ok ? (float)d_mat2_ptr[(size_t)r * cols + gc] : 0.0f;
#endif
			}
		}
		__syncthreads();
		/*//调试代码，出现Bug时选个小矩阵打印一下
		if (tx == 0 && ty == 0) {
			for (int i = 0;i < blockDim.x * 4;i++) {
				for (int j = 0;j < inner_per_times;j++) {
					printf("%0.f", Mat2_Region[i * inner_per_times + j]);
				}
				printf("\n");
			}
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
		const float* TM2BA0 = Mat2_Region + CTXIT;
		const float* TM2BA1 = Mat2_Region + CTXIT + 1 * kInnerPerT;//16=1*blockDim.x
		const float* TM2BA2 = Mat2_Region + CTXIT + 2 * kInnerPerT;
		const float* TM2BA3 = Mat2_Region + CTXIT + 3 * kInnerPerT;
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
			sum00 += r0 * c0; sum01 += r0 * c1; sum02 += r0 * c2, sum03 += r0 * c3,
			sum10 += r1 * c0; sum11 += r1 * c1; sum12 += r1 * c2, sum13 += r1 * c3,
			sum20 += r2 * c0; sum21 += r2 * c1; sum22 += r2 * c2, sum23 += r2 * c3,
			sum30 += r3 * c0; sum31 += r3 * c1; sum32 += r3 * c2, sum33 += r3 * c3;
		}
#endif
		__syncthreads();//再次阻滞，保证每个格子都计算完成再开始下一步
	}
	if (IR0A && IC0A) NBA[0 * cols + 0] = (double)sum00;if (IR0A && IC1A) NBA[0 * cols + 1] = (double)sum01;if (IR0A && IC2A) NBA[0 * cols + 2] = (double)sum02;if (IR0A && IC3A) NBA[0 * cols + 3] = (double)sum03;
	if (IR1A && IC0A) NBA[1 * cols + 0] = (double)sum10;if (IR1A && IC1A) NBA[1 * cols + 1] = (double)sum11;if (IR1A && IC2A) NBA[1 * cols + 2] = (double)sum12;if (IR1A && IC3A) NBA[1 * cols + 3] = (double)sum13;
	if (IR2A && IC0A) NBA[2 * cols + 0] = (double)sum20;if (IR2A && IC1A) NBA[2 * cols + 1] = (double)sum21;if (IR2A && IC2A) NBA[2 * cols + 2] = (double)sum22;if (IR2A && IC3A) NBA[2 * cols + 3] = (double)sum23;
	if (IR3A && IC0A) NBA[3 * cols + 0] = (double)sum30;if (IR3A && IC1A) NBA[3 * cols + 1] = (double)sum31;if (IR3A && IC2A) NBA[3 * cols + 2] = (double)sum32;if (IR3A && IC3A) NBA[3 * cols + 3] = (double)sum33;
	
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
	int gridy = (rows + kTileCols - 1) / (kTileCols);//表示，新得到的矩阵的block有几行
	int gridx = (cols + kTileCols - 1) / (kTileCols);//表示，新得到的矩阵的block有几列
	Matrix_Double_Mutiple_GPU_Kernel << <dim3(gridx, gridy, 1), dim3(kBlockSize, kBlockSize, 1), Shared_Memory_Size >> > (d_mat1_f, d_mat2_f, d_out, rows, inner, cols);
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
void Matrix_Double_Mutiple_GPU(const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out) {
	const int rows = mat1.rowNum_, inner = mat1.colNum_, cols = mat2.colNum_;
	if (inner != mat2.rowNum_) {throw std::runtime_error("mismatched Matrix");}
	if (mat1.data.size() == 0 || mat2.data.size() == 0) {throw std::runtime_error("empty matrix");}
	//结果缓冲复用：仅当尺寸变化时才重新分配；稳态下 resize 是 no-op，既不分配也不清零
	const size_t need = (size_t)rows * cols;
	if (out.data.size() != need) {out.data.resize(need);}
	out.rowNum_ = rows;
	out.colNum_ = cols;

	const size_t newDataSize = need * sizeof(double);
	const size_t mat1Size = mat1.data.size() * sizeof(double);
	const size_t mat2Size = mat2.data.size() * sizeof(double);
	const size_t mat1Sizef = mat1.data.size() * sizeof(float);
	const size_t mat2Sizef = mat2.data.size() * sizeof(float);

	cudaEvent_t e0, e1, e2, e3;
	float ms01 = 0.0, ms12 = 0.0, ms23 = 0.0;
	cudaEventCreate(&e0); cudaEventCreate(&e1);
	cudaEventCreate(&e2); cudaEventCreate(&e3);

	//显存常驻缓冲：容量不足时才重新分配（避免每轮 cudaMalloc/cudaFree）
	if (mat1Size > cap1) {if (d_mat1) {	cudaFree(d_mat1);}CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat1), mat1Size));cap1 = mat1Size;}
	if (mat2Size > cap2) {if (d_mat2) {cudaFree(d_mat2);}CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat2), mat2Size));cap2 = mat2Size;}
	if (newDataSize > capO) {if (d_out) {cudaFree(d_out);}CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), newDataSize));capO = newDataSize;}
	if (mat1Sizef > cap1f) {if (d_mat1_f) {	cudaFree(d_mat1_f);}CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat1_f), mat1Sizef));cap1f = mat1Sizef;}
	if (mat2Sizef > cap2f) {if (d_mat2_f) {cudaFree(d_mat2_f);}CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat2_f), mat2Sizef));cap2f = mat2Sizef;}
	cudaEventRecord(e0);
	CUDA_CHECK(cudaMemcpy(d_mat1, mat1.data.data(), mat1Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_mat2, mat2.data.data(), mat2Size, cudaMemcpyHostToDevice));
	int gridmat1 = (mat1.data.size()+1023) / 1024;
	int gridmat2 = (mat2.data.size()+1023) / 1024;
	DoubleToFloat_Kernel << <gridmat1, 1024 >> > (d_mat1, d_mat1_f, mat1.data.size());
	DoubleToFloat_Kernel << <gridmat2, 1024 >> >(d_mat2, d_mat2_f, mat2.data.size());

	cudaEventRecord(e1);
	const int gridy = (rows + kTileCols - 1) / (kTileCols);//表示，新得到的矩阵的block有几行
	const int gridx = (cols + kTileCols - 1) / (kTileCols);//表示，新得到的矩阵的block有几列
	Matrix_Double_Mutiple_GPU_Kernel << <dim3(gridx, gridy, 1), dim3(kBlockSize, kBlockSize, 1), Shared_Memory_Size >> > (d_mat1_f, d_mat2_f, d_out, rows, inner, cols);
	//诊断打印：只在第一次调用时执行，避免每轮 I/O 污染计时
	
	static bool printed = false;
	if (!printed) {
		printed = true;
		cudaFuncAttributes attr{};
		CUDA_CHECK(cudaFuncGetAttributes(&attr, Matrix_Double_Mutiple_GPU_Kernel));
		int blocksPerSM = 0;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&blocksPerSM, Matrix_Double_Mutiple_GPU_Kernel,
			kBlockSize * kBlockSize, Shared_Memory_Size));
		std::cout << "[ABLATION_MODE=" << ABLATION_MODE << "] regs=" << attr.numRegs
			<< "  shared/block=" << attr.sharedSizeBytes
			<< "  local=" << attr.localSizeBytes << " B"
			<< "  blocks/SM=" << blocksPerSM
			<< "  threads/SM=" << blocksPerSM * kBlockSize * kBlockSize << std::endl;
	}
	
	cudaEventRecord(e2);
	CUDA_CHECK(cudaGetLastError());//调试，看有没有错误信息

	//★ 关键：直接拷回 out 的内部数组 —— 零分配、零清零、零额外拷贝
	CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, newDataSize, cudaMemcpyDeviceToHost));
	cudaEventRecord(e3);
	cudaEventSynchronize(e3);

	cudaEventElapsedTime(&ms01, e0, e1);
	cudaEventElapsedTime(&ms12, e1, e2);
	cudaEventElapsedTime(&ms23, e2, e3);
	std::cout << "[ABLATION_MODE=" << ABLATION_MODE << "] " << ms01 << "ms," << ms12 << "ms," << ms23 << "ms," << std::endl;
}
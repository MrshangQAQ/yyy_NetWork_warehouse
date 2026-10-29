/// ============================================================================
/// 多管道卷积的 GPU 实现（im2col + 瘦 N GEMM）
/// ----------------------------------------------------------------------------
/// 与 CPU 版（Batch_Matrix.cpp 的 GetConvolutionMatrixMul）保持同一套分块逻辑，
/// 只把「展开」和「矩阵乘」两段搬到显存里做，函数签名与调用点零改动。
///
/// 设计要点：
///   1. 计算形状是「瘦 GEMM」：M = n·L 很大（10^4~10^5）、K̄ = C_in·K² 很小（25~400）、
///      N = C_out 只有十几。任何 64 宽的常规 tile 都会在 N 方向浪费 3/4 的算力，
///      因此这里让每个线程负责 A 的一整行、把全部 N 个累加器常驻寄存器，
///      只在 M 方向切条带。
///   2. 精度：主机侧 double → 打包成 float 上传，kernel 全程 FP32 累加，
///      结果拷回后再转回 double。本机 FP64 只有 FP32 的 1/64（约 0.25 TFLOPS，
///      低于 CPU AVX2 的 0.5 TFLOPS），走 double 计算 GPU 必输 CPU。
///   3. 显存缓冲用 static 指针 + 容量变量复用，跨调用不反复 cudaMalloc/cudaFree。
/// ============================================================================
#include "Batch_Matrix.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <chrono>
#include <iomanip>

/// 卷积 GPU 路径的分阶段计时开关：1 = 每次调用打印各阶段耗时；0 = 关闭（零额外开销）。
#define CONV_GPU_TIMING 0

namespace Bat_Convolution_Gpu{
/// 常驻显存缓冲（跨调用复用）。刻意用裸指针而不做析构：
/// 与项目既有做法一致，避免静态对象析构发生在 CUDA 上下文销毁之后。
	float* d_inMat_f = nullptr ; size_t capin = 0;
	float* d_outMat_f = nullptr ; size_t capout = 0;  // 本样本块的输出（n·L 行 × C_out 列）

	float* pinner_in = nullptr ; size_t capPinnerIn = 0;
	float* pinner_out = nullptr ; size_t capPinnerOut = 0;

	constexpr int Shared_Memory_Size = 45056;
	constexpr int ppb = 8;//pipeline_per_block
	constexpr int rpt = 2;//row_per_thread
	constexpr int cpt = 2;//col_per_thread
} // namespace
namespace BCG = Bat_Convolution_Gpu;

/// <summary>GPU 版多管道卷积：GCM(X,Y)[r][c] = Σ_k Conv(X[r][k], Y[k][c], p, s)。</summary>
/// <param name="data">输入图网格+卷积核：其中，输入图在前，卷积核在后</param>
/// <param name="Out_Row">样本数</param>
/// <param name="Out_Col">输出的管道数，即C_Out</param>
/// <param name="Out_Inner">输入的管道数，即C_In，外维乘法被消去的那一维</param>
/// <param name="bat_inner_row">内部图的行数</param>
/// <param name="bat_inner_col">内部图的列数</param>
/// <param name="K">卷积核的直径</param>
/// <param name="padding">四周补零的行列数（与 Convolution 同义）(数据中无p，心中有p(悲))</param>
/// <param name="stride">滑窗步长</param>
/// 规定，y对应行，x对应列
__global__ void Convolution_GPU_kernal(const float* __restrict data,float* __restrict out,int Out_Row,int Out_Col,int Out_inner,int bat_inner_row,int bat_inner_col,int K,int padding,int stride) {
	
	extern __shared__ float Mat_Region[];//存储ker和bat的数据
	/*往这里添加，进行强度削减需要用到的常数*/
	int KK = K * K;//核的元素数

	int total_threads = blockDim.x * blockDim.y;//总线程数
	int thread_Id = threadIdx.y * blockDim.x + threadIdx.x;//该线程的id数

	int out_inner_row = (bat_inner_row + 2 * padding - K) / stride + 1;//输出块矩阵内维的行数
	int out_inner_col = (bat_inner_col + 2 * padding - K) / stride + 1;//输出块矩阵内维的列数


	int tile_col_num = (out_inner_col + 2 * padding + blockDim.x * BCG::cpt - 1) / (blockDim.x * BCG::cpt);//对大矩阵再分块的对于tile块为元素视角下的列数
	//再分块下，该block对应的列索引
	int tile_id_x = blockIdx.z % tile_col_num;
	//再分块下，该block对应的行索引
	int tile_id_y = blockIdx.z / tile_col_num;
	int tile_col_per_block = blockDim.x * BCG::cpt;//再分块下，每个block输出占用的实际列数
	int tile_row_per_block = blockDim.y * BCG::rpt;//再分块下，每个block输出占用的实际行数
	//注，由于我们传入的grid.z是添加了padding的，所以我们再分块下，该block左上角的元素的实际行数和列数要减padding
	int tbc_b = tile_id_x * tile_col_per_block * stride - padding;//该tile左上角的元素在bat中的全局列数
	int tbr_b = tile_id_y * tile_row_per_block * stride - padding;//该tile左上角的元素在bat中的全局行数
	//dataSize系列，表示我们常用的偏移量
	int dataSize_PerBat = bat_inner_row * bat_inner_col;
	int dataSize_PerKerRow = KK * Out_Col;
	int dataSize_PerKerRead = BCG::ppb * KK;
	//读取bat的基地址
	const float* BatReadBaseAddress0 = 
		data
		+ blockIdx.y * Out_inner * dataSize_PerBat//外维偏移,blockIdx.y*gridDim.x表示从外维第blockIdx.y行的元素开始读
		;
	//读取ker的基地址，Out_Row * Out_inner * bat_inner_col * bat_inner_row表示跳过存储bat的部分，blockIdx.x*BCG::ppb表示从第一行的第blockIdx.x*BCG::ppb个Ker开始读
	const float* KerReadBaseAddress0 = data + Out_Row * Out_inner * dataSize_PerBat + blockIdx.x*BCG::ppb*KK;
	//写入out的基地址blockDim
	//实际上，我们写入out中连续的8个矩阵，我们在这里存储第一个矩阵的第一个元素的地址
	//每个out中每个矩阵的大小为out_inner_col*out_inner_row
	//外维偏移由blockIdx决定(blockIdx.x*BCG::ppb+blockIdx.y*gridDim.y)即外维偏移的个数，还要乘以out中每个矩阵的大小
	//内维偏移由本线程决定
	float* OutWriteBaseAddress =
		out
		+ (size_t)(blockIdx.y * Out_Col + blockIdx.x * BCG::ppb)* (size_t)(out_inner_col * out_inner_row)//外维偏移
		+ tile_id_y*tile_row_per_block*out_inner_col+tile_id_x*tile_col_per_block//tile偏移
		;
	///定义计算中的常量，共32个，它们存储在寄存器中
	//填充时规定，相邻的数字通道相邻，通道满8个进列，列满2个进行
	float C[BCG::ppb * BCG::rpt * BCG::cpt] = {};
	///共享内存相关
	//计算时读取bat的基地址
	float* Bat_Region = Mat_Region ;
	int begin_gc0 = threadIdx.x * stride * BCG::cpt;//本线程左上角元素在该bat中对于该block左上脚元素的列全局偏移0
	int begin_gr0 = threadIdx.y * stride * BCG::rpt;//本线程左上角元素在该bat中对于该block左上脚元素的行全局偏移0
	int begin_gc1 = threadIdx.x * stride * BCG::cpt + stride;//本线程左上角元素在该bat中对于该block左上脚元素的列全局偏移1
	int begin_gr1 = threadIdx.y * stride * BCG::rpt + stride;//本线程左上角元素在该bat中对于该block左上脚元素的行全局偏移1
	//这些位置是否可用
	/*
	调试代码
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		printf("%d,%d,%d",begin_gr0, begin_gr0 + threadIdx.y * stride - 1, bat_inner_row);
	}
	*/
	const int orow0 = tile_id_y * tile_row_per_block + threadIdx.y * BCG::rpt;//tile偏移+线程偏移
	const int ocol0 = tile_id_x * tile_col_per_block + threadIdx.x * BCG::cpt;//tile偏移+线程偏移
	const bool w_gr0 = (orow0 + 0) < out_inner_row;
	const bool w_gr1 = (orow0 + 1) < out_inner_row;
	const bool w_gc0 = (ocol0 + 0) < out_inner_col;
	const bool w_gc1 = (ocol0 + 1) < out_inner_col;
	/*该block需要bat内多大的一块空间*/
	//该block需要bat内的空间的列数（偏移量）
	const int bat_inner_col_per_times = tile_col_per_block * stride + K - 1;
	//该block需要bat内的空间的行数（偏移量）
	int bat_inner_row_per_times = tile_row_per_block * stride + K - 1;
	int bat_inner_size_per_times = bat_inner_row_per_times * bat_inner_col_per_times;
	//存储ker的数据,Bat存储在Ker前面
	float* Ker_Region = Mat_Region + bat_inner_size_per_times;//(tile_col_per_block*stride+K-1)*(tile_row_per_block*stride+K-1)为bat的预留空间
	//每个管道中读取的基地址
	float* Ker_Region_Per_Pipepline[BCG::ppb] = 
	{
		Ker_Region + 0 * KK, Ker_Region + 1 * KK, Ker_Region + 2 * KK, Ker_Region + 3 * KK, 
		Ker_Region + 4 * KK, Ker_Region + 5 * KK, Ker_Region + 6 * KK, Ker_Region + 7 * KK
	};
	////开始计算
	for (int k = 0;k < Out_inner;k++) {
		///写入共享内存
		//写入bat,在写入padding的时候越界的自动被拦住了
		const float* BatReadBaseAddress1 = BatReadBaseAddress0 + k * dataSize_PerBat;//本次读取具体是外维这行中的哪一个
		for (int i = thread_Id;i < bat_inner_size_per_times;i+=total_threads) {
			int oc_ = i % bat_inner_col_per_times;//列偏移（快变）
			int or_ = i / bat_inner_col_per_times;//行偏移（慢变）
			int gc_ = tbc_b + i % bat_inner_col_per_times;//全局列号（快变）
			int gr_ = tbr_b + i / bat_inner_col_per_times;//全局行号（慢变）
			Bat_Region[or_*bat_inner_col_per_times+oc_] =(gc_>=0&&gc_<bat_inner_col&& gr_ >= 0 && gr_ < bat_inner_row)?BatReadBaseAddress1[gr_*bat_inner_col+gc_]:0;
		}
		//写入ker，可能出现列越界，需要检查,越界时写入0
		const float* KerReadBaseAddress1 = KerReadBaseAddress0 + k * dataSize_PerKerRow;
		for (int i = thread_Id;i < dataSize_PerKerRead;i += total_threads) {
			Ker_Region[i] =(i/(KK)+blockIdx.x*BCG::ppb<Out_Col) ? KerReadBaseAddress1[i] : 0;
			/*调试代码
			if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
				printf("%f,", KerReadBaseAddress1[i]);
			}
			*/
		}
		__syncthreads();
		/*调试代码
		if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
			printf("ker:\n");
			for (int row = 0;row < K;row++) {
				for (int col = 0;col < K;col++) {
					printf("%.4f ", Ker_Region[row * K + col]);
				}
				printf("\n");
			}
			printf("\n");
		}
			printf("bat:\n");
			for (int row = 0;row < bat_inner_row;row++) {
				for (int col = 0;col < bat_inner_col;col++) {
					printf("%.4f ", Mat_Region[row * bat_inner_col + col]);
				}
				printf("\n");
			}
			printf("\n");
		*/
		///进行计算
		//读取需要的数据
		//读bat中的数据，每次读四个
		for (int OffsetRow = 0;OffsetRow < K; OffsetRow++) {
			int offset_gr0 = begin_gr0 + OffsetRow;
			int offset_gr1 = begin_gr1 + OffsetRow;
			for (int OffsetCol = 0;OffsetCol < K;OffsetCol++) {
				/*读4个需要的bat中的数据*/
				int offset_gc0 = begin_gc0 + OffsetCol;
				int offset_gc1 = begin_gc1 + OffsetCol;
				//bat_data
				float bd[4] = {
				Bat_Region[offset_gr0 * bat_inner_col_per_times + OffsetCol + begin_gc0],
				Bat_Region[offset_gr0 * bat_inner_col_per_times + OffsetCol + begin_gc1],
				Bat_Region[offset_gr1 * bat_inner_col_per_times + OffsetCol + begin_gc0],
				Bat_Region[offset_gr1 * bat_inner_col_per_times + OffsetCol + begin_gc1]
				};
				/*读取8个ker需要的数据*/
				//这里是基础偏移
				int ker_offset = OffsetRow * K + OffsetCol;
				//ker_data,这里kernal在写入的时候，已经保证了越界的部分被写为0
				float kd[8] = {
					Ker_Region_Per_Pipepline[0][ker_offset],
					Ker_Region_Per_Pipepline[1][ker_offset],
					Ker_Region_Per_Pipepline[2][ker_offset],
					Ker_Region_Per_Pipepline[3][ker_offset],
					Ker_Region_Per_Pipepline[4][ker_offset],
					Ker_Region_Per_Pipepline[5][ker_offset],
					Ker_Region_Per_Pipepline[6][ker_offset],
					Ker_Region_Per_Pipepline[7][ker_offset],
				};
				//进行32次计算
				C[0 * BCG::ppb + 0] += kd[0] * bd[0], C[0 * BCG::ppb + 1]+= kd[1] * bd[0], C[0 * BCG::ppb + 2]+= kd[2] * bd[0], C[0 * BCG::ppb + 3]+= kd[3] * bd[0], 
				C[0 * BCG::ppb + 4] += kd[4] * bd[0], C[0 * BCG::ppb + 5]+= kd[5] * bd[0], C[0 * BCG::ppb + 6]+= kd[6] * bd[0], C[0 * BCG::ppb + 7]+= kd[7] * bd[0];
				C[1 * BCG::ppb + 0] += kd[0] * bd[1], C[1 * BCG::ppb + 1]+= kd[1] * bd[1], C[1 * BCG::ppb + 2]+= kd[2] * bd[1], C[1 * BCG::ppb + 3]+= kd[3] * bd[1],
				C[1 * BCG::ppb + 4] += kd[4] * bd[1], C[1 * BCG::ppb + 5]+= kd[5] * bd[1], C[1 * BCG::ppb + 6]+= kd[6] * bd[1], C[1 * BCG::ppb + 7]+= kd[7] * bd[1];
				C[2 * BCG::ppb + 0] += kd[0] * bd[2], C[2 * BCG::ppb + 1]+= kd[1] * bd[2], C[2 * BCG::ppb + 2]+= kd[2] * bd[2], C[2 * BCG::ppb + 3]+= kd[3] * bd[2],
				C[2 * BCG::ppb + 4] += kd[4] * bd[2], C[2 * BCG::ppb + 5]+= kd[5] * bd[2], C[2 * BCG::ppb + 6]+= kd[6] * bd[2], C[2 * BCG::ppb + 7]+= kd[7] * bd[2];
				C[3 * BCG::ppb + 0] += kd[0] * bd[3], C[3 * BCG::ppb + 1]+= kd[1] * bd[3], C[3 * BCG::ppb + 2]+= kd[2] * bd[3], C[3 * BCG::ppb + 3]+= kd[3] * bd[3],
				C[3 * BCG::ppb + 4] += kd[4] * bd[3], C[3 * BCG::ppb + 5]+= kd[5] * bd[3], C[3 * BCG::ppb + 6]+= kd[6] * bd[3], C[3 * BCG::ppb + 7]+= kd[7] * bd[3];
				/*调试代码
				if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
					printf("kd:%f,bd:%f,C:%f\n", kd[0], bd[0], C[1]);
				}
				*/
			}
		}
		//进入下一轮
		__syncthreads();
	}
	///把结果写入out中,这里要注意Ker访问越界和位置访问越界
	int real_pipepline_num = Out_Col - blockIdx.x * BCG::ppb < BCG::ppb ? Out_Col - blockIdx.x * BCG::ppb : BCG::ppb;
	for (int pipepline = 0;pipepline < real_pipepline_num ; pipepline++) {
		float* OutWrite_ptr = OutWriteBaseAddress + pipepline * out_inner_col * out_inner_row;
		if (w_gr0 && w_gc0) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt)]				      = C[0 * BCG::ppb + pipepline]; }
		if (w_gr0 && w_gc1) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt) + 1]			      = C[1 * BCG::ppb + pipepline]; }
		if (w_gr1 && w_gc0) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt) + out_inner_col ]    = C[2 * BCG::ppb + pipepline]; }
		if (w_gr1 && w_gc1) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt) + out_inner_col + 1] = C[3 * BCG::ppb + pipepline]; }
	}
}
//用于辅助写入锁页内存
static void PackD2F(const double* __restrict src, float* __restrict dst, size_t n) {
	size_t i = 0;
	for (; i + 4 <= n; i += 4) {
		__m256d d = _mm256_loadu_pd(src + i);   // 读 4 个 double
		__m128  f = _mm256_cvtpd_ps(d);         // 转成 4 个 float
		_mm_storeu_ps(dst + i, f);              // 写 4 个 float
	}
	for (; i < n; ++i) dst[i] = (float)src[i];  // 余数
}
static void PackF2D(const float* __restrict src, double* __restrict dst, size_t n) {
	size_t i = 0;
	for (; i + 4 <= n; i += 4) {
		__m128  f = _mm_loadu_ps(src + i);
		__m256d d = _mm256_cvtps_pd(f);
		_mm256_storeu_pd(dst + i, d);
	}
	for (; i < n; ++i) dst[i] = (double)src[i];
}
void GetConvolutionMatrixMul_GPU(const Batch_Matrix& bat, const Batch_Matrix& ker,Batch_Matrix& out, int padding, int stride)
{
	// ---- 入口校验（与 CPU 版保持一致）----
	if (bat.GetColNum() != ker.GetRowNum()) {
		throw std::runtime_error("mismatched bat");
	}
	const std::vector<Matrix_Double>& kd = ker.GetInternalData();
	if (kd[0].GetRowNum() != kd[0].GetColNum()) {
		throw std::runtime_error("the ker couldn't serve as the kernel");
	}
	// ---- 分阶段计时：CPU 侧 steady_clock，GPU 侧 cudaEvent（与 Matrix_Double_Mutiple_GPU 同一套做法）----
#if CONV_GPU_TIMING
	using SteadyClock = std::chrono::steady_clock;
	const auto t_begin = SteadyClock::now();
	static cudaEvent_t eH2D0 = nullptr, eH2D1 = nullptr, eKernel = nullptr, eD2H = nullptr;
	if (eH2D0 == nullptr) {
		CUDA_CHECK(cudaEventCreate(&eH2D0));
		CUDA_CHECK(cudaEventCreate(&eH2D1));
		CUDA_CHECK(cudaEventCreate(&eKernel));
		CUDA_CHECK(cudaEventCreate(&eD2H));
	}
#endif
	/*分配空间预处理*/
	size_t batTotalSize = bat.GetInternalData().size() * bat.GetInternalData()[0].GetInternalData().size();
	size_t kerTotalSize = ker.GetInternalData().size() * ker.GetInternalData()[0].GetInternalData().size();
	if (BCG::capPinnerIn < batTotalSize + kerTotalSize) {
		if (BCG::pinner_in) {cudaFreeHost(BCG::pinner_in);}
		CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&BCG::pinner_in), (batTotalSize + kerTotalSize) * sizeof(float)));
		BCG::capPinnerIn = batTotalSize + kerTotalSize;
	}
	if (BCG::capin < batTotalSize + kerTotalSize) {
		if (BCG::d_inMat_f) { cudaFree(BCG::d_inMat_f); }
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&BCG::d_inMat_f), (batTotalSize + kerTotalSize) * sizeof(float)));
		BCG::capin = batTotalSize + kerTotalSize;
	}
	const int R = bat.GetRowNum();      // bat 的行数（前向时 = 批大小；反传 dK 时 = 层通道数）,即样本数
	const int C_in = bat.GetColNum();   // bat 的列数（前向时 = 输入通道数；反传 dK 时 = 批大小）
	const int C_out = ker.GetColNum();  // 输出通道数
	const int K = ker.GetInternalData()[0].GetRowNum();
	const int KK = K * K;

	//bat经过padding后的内维大小
	const int H = bat.GetInternalData()[0].GetRowNum()+2*padding;
	const int W = bat.GetInternalData()[0].GetColNum()+2*padding;

	/*out的内维大小*/
	//out的内维行数
	const int out_inner_Rows = (H - K) / stride + 1;   
	//out的内维列数
	const int out_inner_Cols = (W - K) / stride + 1;

	size_t outTotalSize = (size_t)out_inner_Cols * out_inner_Rows * R * C_out;
	if (BCG::capPinnerOut < outTotalSize) {
		if (BCG::pinner_out) { cudaFreeHost(BCG::pinner_out); }
		CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&BCG::pinner_out), outTotalSize * sizeof(float)));
		BCG::capPinnerOut = outTotalSize;
	}
	if (BCG::capout < outTotalSize) {
		if (BCG::d_outMat_f) { cudaFree(BCG::d_outMat_f); }
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&BCG::d_outMat_f), outTotalSize * sizeof(float)));
		BCG::capout = outTotalSize;
	}
	/*将数据写入锁页内存中*/
	std::vector<float*> write_ptrs_bat(bat.GetInternalData().size());
	std::vector<float*> write_ptrs_ker(ker.GetInternalData().size());
#pragma omp parallel for if(bat.GetInternalData().size()>64)
	for (int i = 0;i < bat.GetInternalData().size();i++) {
		write_ptrs_bat[i] = BCG::pinner_in + i * bat.GetInternalData()[0].GetInternalData().size();
		PackD2F(bat.GetInternalData()[i].GetInternalData().data(), write_ptrs_bat[i], bat.GetInternalData()[i].GetInternalData().size());
	}
#pragma omp parallel for if(ker.GetInternalData().size()>64)
	for (int i = 0;i < ker.GetInternalData().size();i++) {
		write_ptrs_ker[i] = BCG::pinner_in + batTotalSize + i * ker.GetInternalData()[0].GetInternalData().size();
		PackD2F(ker.GetInternalData()[i].GetInternalData().data(), write_ptrs_ker[i], ker.GetInternalData()[i].GetInternalData().size());
	}
	//拷贝进入显存中
#if CONV_GPU_TIMING
	const auto t_packed = SteadyClock::now();
	cudaEventRecord(eH2D0);
#endif
	CUDA_CHECK(cudaMemcpy(BCG::d_inMat_f, BCG::pinner_in, (batTotalSize + kerTotalSize) * sizeof(float), cudaMemcpyHostToDevice));
#if CONV_GPU_TIMING
	cudaEventRecord(eH2D1);
#endif
	/*计算部分*/
	//这里的规划部分，可以考虑做一个函数arrangement，返回一个结构体
	//每一个block对应8个通道，每个线程在每个通道有2*2个输出所以大概应该这样
	int gridy = R;
	int gridx = (C_out + BCG::ppb-1) / BCG::ppb;
	int gridz = 1 ;//表示这8个通道由几个block共同完成(bat内再分块，每个block负责自己的一块
	int blocky = (out_inner_Rows + BCG::rpt - 1) / BCG::rpt;
	int blockx = (out_inner_Cols + BCG::cpt - 1) / BCG::cpt;

	// 本 tile 实际需要的 shared（float 个数）—— 与 kernel 里的计算完全一致
	auto tileFloats = [&](int bx, int by) -> size_t {
		const size_t wIn = (size_t)bx * BCG::cpt * stride + K - 1;
		const size_t hIn = (size_t)by * BCG::rpt * stride + K - 1;
		return hIn * wIn + (size_t)BCG::ppb * KK;
		};

	// 放不下 / 线程超 1024 就缩 tile（同时满足两个约束）
	while (
		(blockx > 1 || blocky > 1) &&
		(tileFloats(blockx, blocky) * sizeof(float) > (size_t)BCG::Shared_Memory_Size ||
		blockx * blocky > 1024)
		) 
	{
		if (blockx >= blocky && blockx > 1) --blockx;
		else if (blocky > 1)                --blocky;
	}
	if (tileFloats(blockx, blocky) * sizeof(float) > (size_t)BCG::Shared_Memory_Size)
		throw std::runtime_error("shared memory too small for one output tile");
	if (blockx * blocky > 1024)
		throw std::runtime_error("too many threads per block");
	int Real_Shared_Size = (int)(tileFloats(blockx, blocky) * sizeof(float));

	// 只有确实要分块时才需要 z 维
	if (blockx * BCG::cpt < out_inner_Cols || blocky * BCG::rpt < out_inner_Rows) {
		gridz = ((out_inner_Cols + blockx * BCG::cpt - 1) / (blockx * BCG::cpt))
			* ((out_inner_Rows + blocky * BCG::rpt - 1) / (blocky * BCG::rpt));
	}

#if CONV_GPU_TIMING
	// 首次调用打印一次资源配置（寄存器 / 占用率），便于与基线对比
	static bool sCfgPrinted = false;
	if (!sCfgPrinted) {
		sCfgPrinted = true;
		cudaFuncAttributes attr{};
		CUDA_CHECK(cudaFuncGetAttributes(&attr, Convolution_GPU_kernal));
		int blocksPerSM = 0;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&blocksPerSM, Convolution_GPU_kernal, blockx * blocky, Real_Shared_Size));
		std::cout << "[conv-GPU cfg] regs=" << attr.numRegs
			<< " local=" << attr.localSizeBytes << "B"
			<< " blocks/SM=" << blocksPerSM
			<< " threads/SM=" << blocksPerSM * blockx * blocky
			<< " warps/SM=" << (blocksPerSM * blockx * blocky) / 32 << std::endl;
	}
#endif

	Convolution_GPU_kernal
		<<<dim3(gridx, gridy, gridz), dim3(blockx, blocky, 1), Real_Shared_Size >>>
		(BCG::d_inMat_f, BCG::d_outMat_f, R, C_out, C_in, bat.GetInternalData()[0].GetRowNum(), bat.GetInternalData()[0].GetColNum(), K, padding, stride);
	CUDA_CHECK(cudaGetLastError());
#if CONV_GPU_TIMING
	cudaEventRecord(eKernel);
#endif
	// ---- 回传：整批一次 D2H ----
	CUDA_CHECK(cudaMemcpy(BCG::pinner_out, BCG::d_outMat_f, outTotalSize * sizeof(float), cudaMemcpyDeviceToHost));
#if CONV_GPU_TIMING
	cudaEventRecord(eD2H);
	cudaEventSynchronize(eD2H);
	const auto t_gpu_done = SteadyClock::now();
#endif

	// 输出布局与 Batch_Matrix 的拍平顺序一致：按块一次线性解包即可
	const bool dimsMatch =
		out.data.size() != 0 &&
		out.GetColNum() == C_out && out.GetRowNum() == R &&
		(int)out.GetInternalData().size() == R * C_out &&
		out.GetInternalData()[0].GetColNum() == out_inner_Cols &&
		out.GetInternalData()[0].GetRowNum() == out_inner_Rows;
	if (!dimsMatch) {
		out.rowNums_ = R;
		out.colNums_ = C_out;
		out.data.assign((size_t)R * C_out, Matrix_Double(out_inner_Rows, out_inner_Cols));
	}

	const size_t outPerBlock = (size_t)out_inner_Rows * out_inner_Cols;
#pragma omp parallel for if(R * C_out > 64)
	for (int i = 0; i < R * C_out; ++i) {
		PackF2D(BCG::pinner_out + (size_t)i * outPerBlock,
			Internal::Matrix_Double_Access::GetInterData(out.data[i]).data(), outPerBlock);
	}

#if CONV_GPU_TIMING
	const auto t_end = SteadyClock::now();
	const double msPack = std::chrono::duration<double, std::milli>(t_packed - t_begin).count();
	const double msUnpack = std::chrono::duration<double, std::milli>(t_end - t_gpu_done).count();
	const double msTotal = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
	float msH2D = 0.0f, msKernel = 0.0f, msD2H = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&msH2D, eH2D0, eH2D1));
	CUDA_CHECK(cudaEventElapsedTime(&msKernel, eH2D1, eKernel));
	CUDA_CHECK(cudaEventElapsedTime(&msD2H, eKernel, eD2H));
	const auto oldFlags = std::cout.flags();
	const auto oldPrec = std::cout.precision();
	std::cout << std::fixed << std::setprecision(3)
		<< "[conv-GPU] grid=(" << gridx << ',' << gridy << ',' << gridz << ')'
		<< " block=(" << blockx << ',' << blocky << ')'
		<< " shmem=" << Real_Shared_Size << "B"
		<< " | pack=" << msPack << " H2D=" << msH2D << " kernel=" << msKernel
		<< " D2H=" << msD2H << " unpack=" << msUnpack << std::endl 
		<< " total=" << msTotal << " ms" << std::endl;
	std::cout.flags(oldFlags);
	std::cout.precision(oldPrec);
#endif
}
/// <summary>估算 GPU 卷积会开出的总线程数：gridx × R × ceil(outCols/cpt) × ceil(outRows/rpt)。</summary>
/// <remarks>形状太小时 GPU 只有几十个线程，反而不如 CPU；自适应调度用它做第二道判据。</remarks>
size_t GetConvGpuThreadCount(int R, int C_out, int outRows, int outCols) {
	const int gridx = (C_out + BCG::ppb - 1) / BCG::ppb;
	const int blockx = (outCols + BCG::cpt - 1) / BCG::cpt;
	const int blocky = (outRows + BCG::rpt - 1) / BCG::rpt;
	return (size_t)(gridx > 0 ? gridx : 1) * (size_t)(R > 0 ? R : 1) * (size_t)blockx * (size_t)blocky;
}
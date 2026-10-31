#include"Batch_Matrix_GPU.h"
#include <immintrin.h>
namespace Bat_Convolution_Gpu {
	float* d_inMat_f = nullptr; size_t capin = 0;
	float* d_outMat_f = nullptr; size_t capout = 0;
	float* pinner_in = nullptr; size_t capPinnerIn = 0;
	float* pinner_out = nullptr; size_t capPinnerOut = 0;
}
//将data1和data2两个地址后的接下来Addsize个浮点数相加，然后放到out中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerAddTo(const float* const __restrict data1, const float* const __restrict data2, float* __restrict out, size_t AddSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < AddSize;i += totalThreads) {
		out[i] = data1[i] + data2[i];
	}
}
//将data1和data2两个地址后的接下来Addsize个浮点数相加，然后放到data1中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerAdd(float* const __restrict data1, const float* const __restrict data2, size_t AddSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < AddSize;i += totalThreads) {
		data1[i]+=data2[i];
	}
}
//将data1和data2两个地址后的接下来Addsize个浮点数相减，然后放到out中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerMinTo(const float* const __restrict data1, const float* const __restrict data2, float* __restrict out, size_t MinSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < MinSize;i += totalThreads) {
		out[i] = data1[i] - data2[i];
	}
}
//将data1和data2两个地址后的接下来Addsize个浮点数相减少，然后放到data1中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerMin(float* const __restrict data1, const float* const __restrict data2, size_t MinSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < MinSize;i +=totalThreads) {
		data1[i] -= data2[i];
	}
}
//将data1和data2两个地址后的接下来Addsize个浮点数相乘，然后放到out中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerMulTo(const float* const __restrict data1, const float* const __restrict data2, float* __restrict out, size_t MulSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < MulSize;i +=totalThreads) {
		out[i] = data1[i] * data2[i];
	}
}
//将data1和data2两个地址后的接下来Addsize个浮点数相乘，然后放到data1中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerMul(float* const __restrict data1, const float* const __restrict data2, size_t MulSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < MulSize;i +=totalThreads) {
		data1[i] *= data2[i];
	}
}
//将data两个地址后的接下来Addsize个浮点数每个都乘以f，然后放到out中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerMulTo(const float* const __restrict data, float f, float* const __restrict out, size_t MulSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < MulSize;i +=totalThreads) {
		out[i] = data[i]*f;
	}
}
//将data1两个地址后的接下来Addsize个浮点数每个都乘以f，然后放到data中
//确定线程数时建议除x维外，其余两维均为1
static __global__ void PerMul(float* const __restrict data, float f, size_t MulSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < MulSize;i +=totalThreads) {
		data[i] *= f;
	}
}
//将data中的每个元素都作用f一次
//确定线程数时建议除x维外，其余两维均为1
template<typename F>
static __global__ void PerMap(float* const __restrict data, F f, size_t MapSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	for (int i = id;i < MapSize;i += totalThreads) {
		data[i] = f(data[i]);
	}
}
/// <summary>
/// 每个block负责一个内维矩阵，将内维矩阵旋转180度
/// 确定线程数时建议threads除x维外，其余两维均为1,block可以有y维，z维为1
/// </summary>
/// <returns></returns>
static __global__ void GPU_InterRotato_180(const float* const data,float* const out, size_t outrownums, size_t outcolnums, size_t inrownums, size_t incolnums) {
	size_t sizeper_inmat = inrownums * incolnums;
	size_t id_block = blockIdx.y * gridDim.x + blockIdx.x;
	size_t id_threads = threadIdx.x;
	size_t totalThreads_perBlock = blockDim.x;
	const float* BeginAddress = data + sizeper_inmat * id_block;
	float* EndAddress = out + sizeper_inmat * ( 1 + id_block ) - 1;
	float tmp = 0.0;
	for (int i = id_threads;i < sizeper_inmat;i += totalThreads_perBlock) {
		EndAddress[-i] = BeginAddress[i];

	}
}
/// <summary>
/// 每个block负责一个内维矩阵，将内维矩阵按照pollsize池化
/// 确定线程数时建议threads有x维与y维,block有x维与y维，z维都为1
/// 每个线程写入一个数据，负责一个pollsize*pollsize的区域
/// </summary>
/// <returns></returns>
static __global__ void GPU_MaxPoll(const float* const data,float* const pollBuffer,size_t* antiPollBuffer, size_t outrownums, size_t outcolnums, size_t inrownums, size_t incolnums,size_t pollsize) {
	size_t sizePerInMat = inrownums * incolnums;
	size_t id_block = blockIdx.y * gridDim.x + blockIdx.x;
	size_t id_threads = threadIdx.x + threadIdx.y * blockDim.x;
	size_t totalThreads_perBlock = blockDim.x * blockDim.y;
	//计算该块负责的矩阵的基址
	const float* BaseAddress0 = data + sizePerInMat * id_block;

	size_t newInRowNums = inrownums / pollsize;//得到的池化后内部矩阵的行数
	size_t newInColNums = incolnums / pollsize;//得到的池化后内部矩阵的列数
	size_t newSizePerInMat = newInRowNums * newInColNums;

	//该块写入池化缓冲区的基地址
	float* PollWriteAddress0 = pollBuffer + id_block * newSizePerInMat;

	//该块写入池化后得到的反池化数据的基地址
	//空出三格空间，依次存储pollsize，原内部矩阵的列数和行数
	size_t* AntiPollWriteAddress0 = antiPollBuffer + 3 + id_block * newSizePerInMat ;
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		antiPollBuffer[0] = pollsize;
		antiPollBuffer[1] = incolnums;
		antiPollBuffer[2] = inrownums;
	}

	//计算该线程负责的矩阵的基地址
	float max = -100;
	int maxC = 0;
	int maxR = 0;
	const float* BaseAddress1 = BaseAddress0 + pollsize * threadIdx.x + pollsize * threadIdx.y * incolnums;
	for (int offsetRow = 0;offsetRow < pollsize;offsetRow++) {
		const float* BaseAddress2 = BaseAddress1 + offsetRow * incolnums;
		for (int offsetCol = 0;offsetCol < pollsize;offsetCol++) {
			if (max < BaseAddress2[offsetCol]) {
				max = BaseAddress2[offsetCol];
				maxC = offsetCol;
				maxR = offsetRow;
			}
		}
	}
	//计算完毕，写入数据
	AntiPollWriteAddress0[id_threads] = maxR * pollsize + maxC;
	PollWriteAddress0[id_threads] = max;
	//printf("polling end:tx:%d,ty:%d,dx:%d,dy:%d.  write:%f\n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y, PollWriteAddress0[id_threads]);
}
/// <summary>
/// 一键反池化乘以常数f后加到data中去
/// 线程数建议y，z维为1，仅含有x维，block数建议为2维，x/y维分别为data对应矩阵的外维列数和行数
/// </summary>
/// <returns></returns>
static __global__ void GPU_AntiPoll_Timesf_Addto(const float* const pollBuffer, const size_t* const antiPollBuffer,float* data, float f,size_t inrownums ,size_t incolnums) {
	size_t pollsize = antiPollBuffer[0];

	size_t inColNums = inrownums;
	size_t inRowNums = incolnums;
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		if (inColNums != antiPollBuffer[1] || inRowNums != antiPollBuffer[2]) {
			printf("warning!the anti pollBuffer mismatch with the bat!Please confirm if this is your intention");
		}
	}
	size_t sizePerMat = inColNums * inRowNums;

	size_t pollInColNums = inColNums / pollsize;
	size_t pollInRowNums = inRowNums / pollsize;
	size_t pollSizePerInMat = pollInColNums * pollInRowNums;

	size_t id_block = blockIdx.y * gridDim.x + blockIdx.x;
	size_t id_threads = threadIdx.x;
	size_t totalThreads = blockDim.x;
	//读取poll中数据的基地址
	const float* pollReadBaseAddress = pollBuffer + id_block * pollSizePerInMat;
	//读取antipoll中数据的基地址
	const size_t* antiPollReadBaseAddress = antiPollBuffer + id_block * pollSizePerInMat + 3;
	//写入data中数据的基地址
	float* dataWriteBaseAddress = data + sizePerMat * id_block;
	//开始填入数据
	for (int i = id_threads;i < pollSizePerInMat;i += totalThreads) {
		int baseRow = pollsize * (i / pollInColNums);
		int baseCol = pollsize * (i % pollInColNums);
		int offsetRow = antiPollReadBaseAddress[i] / pollsize;
		int offsetCol = antiPollReadBaseAddress[i] % pollsize;
		dataWriteBaseAddress[(baseRow + offsetRow) * inColNums + (offsetCol + baseCol)] += f * pollReadBaseAddress[i];
	}
}
/// <summary>
/// 在data内部矩阵相邻的每个元素之间插入stride个0，将之填入out中
/// 规定block为2维，x/y维分别代表外维的列数与行数，线程数建议1维
/// setzero用于加速，如果Out复用，stride不变，不需要在其余位置重复填0，只用改变目标位置的值即可，设置setzero为flase以跳过重复的填0
/// </summary>
/// <returns></returns>
static __global__ void GPU_Dilate(const float* const data,float* const out,size_t stride,size_t inrownums,size_t incolnums,bool setzero) {
	size_t id_block = blockIdx.y * gridDim.x + blockIdx.x;
	size_t id_threads = threadIdx.x;
	size_t totalThreads = blockDim.x;

	size_t dataSizePerInMat = incolnums * inrownums;

	size_t outInColNums = (incolnums-1) * stride + 1;
	size_t outInRowNums = (inrownums-1)  * stride + 1;
	size_t outSizePerInMat = outInColNums * outInRowNums;
	//该块从data中读取数据的基地址
	const float* dataReadBaseAddress = data + dataSizePerInMat * id_block;
	//该块写入out的基地址
	float* outWriteBaseAddress = out + outSizePerInMat * id_block ;
	//若需要setzero
	if (setzero) {
		for (int i = id_threads;i < outSizePerInMat;i += totalThreads) {
			outWriteBaseAddress[i] = 0.0f;
		}
		__syncthreads();
	}
	for (int i = id_threads;i < dataSizePerInMat;i += totalThreads) {
		int outRow = stride * ( i / incolnums);
		int outcol = stride * ( i % incolnums);
		outWriteBaseAddress[outRow * outInColNums + outcol] = dataReadBaseAddress[i];
	}
}
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
static __global__ void Convolution_GPU_kernal(
	const float* const __restrict bat, const float* const __restrict ker, float* __restrict const out, //数据空间
	size_t Out_Row, size_t Out_Col, size_t Out_inner, size_t bat_inner_row, size_t bat_inner_col, size_t K, //bat与Ker相关参数
	size_t padding, size_t stride//卷积的额外参数
) {

	extern __shared__ float Mat_Region[];//存储ker和bat的数据
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
		bat
		+ blockIdx.y * Out_inner * dataSize_PerBat//外维偏移,blockIdx.y*gridDim.x表示从外维第blockIdx.y行的元素开始读
		;
	//读取ker的基地址blockIdx.x*BCG::ppb表示从第一行的第blockIdx.x*BCG::ppb个Ker开始读
	const float* KerReadBaseAddress0 = ker + blockIdx.x * BCG::ppb * KK;
	//写入out的基地址blockDim
	//实际上，我们写入out中连续的8个矩阵，我们在这里存储第一个矩阵的第一个元素的地址
	//每个out中每个矩阵的大小为out_inner_col*out_inner_row
	//外维偏移由blockIdx决定(blockIdx.x*BCG::ppb+blockIdx.y*gridDim.y)即外维偏移的个数，还要乘以out中每个矩阵的大小
	//内维偏移由本线程决定
	float* OutWriteBaseAddress =
		out
		+ (size_t)(blockIdx.y * Out_Col + blockIdx.x * BCG::ppb) * (size_t)(out_inner_col * out_inner_row)//外维偏移
		+ tile_id_y * tile_row_per_block * out_inner_col + tile_id_x * tile_col_per_block//tile偏移
		;
	///定义计算中的常量，共32个，它们存储在寄存器中
	//填充时规定，相邻的数字通道相邻，通道满8个进列，列满2个进行
	float C[BCG::ppb * BCG::rpt * BCG::cpt] = {};
	///共享内存相关
	//计算时读取bat的基地址
	float* Bat_Region = Mat_Region;
	int begin_gc0 = threadIdx.x * stride * BCG::cpt;//本线程左上角元素在该bat中对于该block左上脚元素的列全局偏移0
	int begin_gr0 = threadIdx.y * stride * BCG::rpt;//本线程左上角元素在该bat中对于该block左上脚元素的行全局偏移0
	int begin_gc1 = threadIdx.x * stride * BCG::cpt + stride;//本线程左上角元素在该bat中对于该block左上脚元素的列全局偏移1
	int begin_gr1 = threadIdx.y * stride * BCG::rpt + stride;//本线程左上角元素在该bat中对于该block左上脚元素的行全局偏移1
	//这些位置是否可用
	const int orow0 = tile_id_y * tile_row_per_block + threadIdx.y * BCG::rpt;//tile偏移+线程偏移
	const int ocol0 = tile_id_x * tile_col_per_block + threadIdx.x * BCG::cpt;//tile偏移+线程偏移
	const bool w_gr0 = (orow0 + 0) < out_inner_row;
	const bool w_gr1 = (orow0 + 1) < out_inner_row;
	const bool w_gc0 = (ocol0 + 0) < out_inner_col;
	const bool w_gc1 = (ocol0 + 1) < out_inner_col;
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
		for (int i = thread_Id;i < bat_inner_size_per_times;i += total_threads) {
			int oc_ = i % bat_inner_col_per_times;//列偏移（快变）
			int or_ = i / bat_inner_col_per_times;//行偏移（慢变）
			int gc_ = tbc_b + i % bat_inner_col_per_times;//全局列号（快变）
			int gr_ = tbr_b + i / bat_inner_col_per_times;//全局行号（慢变）
			Bat_Region[or_ * bat_inner_col_per_times + oc_] = (gc_ >= 0 && gc_ < bat_inner_col && gr_ >= 0 && gr_ < bat_inner_row) ? BatReadBaseAddress1[gr_ * bat_inner_col + gc_] : 0;
		}
		//写入ker，可能出现列越界，需要检查,越界时写入0
		const float* KerReadBaseAddress1 = KerReadBaseAddress0 + k * dataSize_PerKerRow;
		for (int i = thread_Id;i < dataSize_PerKerRead;i += total_threads) {
			Ker_Region[i] = (i / (KK)+blockIdx.x * BCG::ppb < Out_Col) ? KerReadBaseAddress1[i] : 0;
		}
		__syncthreads();
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
				C[0 * BCG::ppb + 0] += kd[0] * bd[0], C[0 * BCG::ppb + 1] += kd[1] * bd[0], C[0 * BCG::ppb + 2] += kd[2] * bd[0], C[0 * BCG::ppb + 3] += kd[3] * bd[0],
					C[0 * BCG::ppb + 4] += kd[4] * bd[0], C[0 * BCG::ppb + 5] += kd[5] * bd[0], C[0 * BCG::ppb + 6] += kd[6] * bd[0], C[0 * BCG::ppb + 7] += kd[7] * bd[0];
				C[1 * BCG::ppb + 0] += kd[0] * bd[1], C[1 * BCG::ppb + 1] += kd[1] * bd[1], C[1 * BCG::ppb + 2] += kd[2] * bd[1], C[1 * BCG::ppb + 3] += kd[3] * bd[1],
					C[1 * BCG::ppb + 4] += kd[4] * bd[1], C[1 * BCG::ppb + 5] += kd[5] * bd[1], C[1 * BCG::ppb + 6] += kd[6] * bd[1], C[1 * BCG::ppb + 7] += kd[7] * bd[1];
				C[2 * BCG::ppb + 0] += kd[0] * bd[2], C[2 * BCG::ppb + 1] += kd[1] * bd[2], C[2 * BCG::ppb + 2] += kd[2] * bd[2], C[2 * BCG::ppb + 3] += kd[3] * bd[2],
					C[2 * BCG::ppb + 4] += kd[4] * bd[2], C[2 * BCG::ppb + 5] += kd[5] * bd[2], C[2 * BCG::ppb + 6] += kd[6] * bd[2], C[2 * BCG::ppb + 7] += kd[7] * bd[2];
				C[3 * BCG::ppb + 0] += kd[0] * bd[3], C[3 * BCG::ppb + 1] += kd[1] * bd[3], C[3 * BCG::ppb + 2] += kd[2] * bd[3], C[3 * BCG::ppb + 3] += kd[3] * bd[3],
					C[3 * BCG::ppb + 4] += kd[4] * bd[3], C[3 * BCG::ppb + 5] += kd[5] * bd[3], C[3 * BCG::ppb + 6] += kd[6] * bd[3], C[3 * BCG::ppb + 7] += kd[7] * bd[3];
			}
		}
		//进入下一轮
		__syncthreads();
	}
	///把结果写入out中,这里要注意Ker访问越界和位置访问越界
	int real_pipepline_num = Out_Col - blockIdx.x * BCG::ppb < BCG::ppb ? Out_Col - blockIdx.x * BCG::ppb : BCG::ppb;
	for (int pipepline = 0;pipepline < real_pipepline_num; pipepline++) {
		float* OutWrite_ptr = OutWriteBaseAddress + pipepline * out_inner_col * out_inner_row;
		if (w_gr0 && w_gc0) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt)] = C[0 * BCG::ppb + pipepline]; }
		if (w_gr0 && w_gc1) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt) + 1] = C[1 * BCG::ppb + pipepline]; }
		if (w_gr1 && w_gc0) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt) + out_inner_col] = C[2 * BCG::ppb + pipepline]; }
		if (w_gr1 && w_gc1) { OutWrite_ptr[(threadIdx.y * BCG::rpt * out_inner_col + threadIdx.x * BCG::cpt) + out_inner_col + 1] = C[3 * BCG::ppb + pipepline]; }
	}
}



GPUBatchMatrix& operator += (GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2) {
	if (!IsMatch(bat1, bat2)) {
		throw(std::runtime_error("mismatched batch_matrix couldn't add"));
	}
	size_t Size = bat1.GetTotalSize();
	size_t blockNums = ( Size +( 1024*64 )-1 )/ (1024 * 64);
	size_t threadNums = 1024;
	PerAdd << <dim3(blockNums, 1, 1), dim3(threadNums, 1, 1) >> > (bat1.GetData(), bat2.GetDataOnlyRead(),Size);
	CUDA_CHECK(cudaGetLastError());
	return bat1;
}
GPUBatchMatrix& operator -= (GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2) {
	if (!IsMatch(bat1, bat2)) {
		throw(std::runtime_error("mismatched batch_matrix couldn't min"));
	}
	size_t Size = bat1.GetTotalSize();
	size_t blockNums = (Size + (1024 * 64) - 1) / (1024 * 64);
	size_t threadNums = 1024;
	PerMin << <dim3(blockNums, 1, 1), dim3(threadNums, 1, 1) >> > (bat1.GetData(), bat2.GetDataOnlyRead(), Size);
	CUDA_CHECK(cudaGetLastError());
	return bat1;
}
GPUBatchMatrix& operator *= (GPUBatchMatrix& bat1, float f) {
	size_t Size = bat1.GetTotalSize();
	size_t blockNums = (Size + (1024 * 64) - 1) / (1024 * 64);
	size_t threadNums = 1024;
	PerMul << <dim3(blockNums, 1, 1), dim3(threadNums, 1, 1) >> > (bat1.GetData(), f, Size);
	CUDA_CHECK(cudaGetLastError());
	return bat1;
}
//反转180度，反向传播时用
void InterRotato_180(const GPUBatchMatrix& bat, GPUBatchMatrix& out) {
	if (!out.TryMatchWith(bat.GetOutRowNums(),bat.GetOutColNums(),bat.GetInRowNums(),bat.GetInColNums())) {
		throw std::runtime_error("Calibration out failed in InterRotato_180");
	}
	size_t threadNums = bat.GetInSize() / 64 > 1024 ? bat.GetInSize() / 64 : 1024;
	GPU_InterRotato_180<<<dim3(bat.GetOutColNums(),bat.GetOutRowNums(),1),dim3(threadNums,1,1) >> >(
		bat.GetDataOnlyRead(),out.GetData(),bat.GetOutRowNums(),bat.GetOutColNums(),bat.GetInRowNums(),bat.GetInColNums()
	);
	CUDA_CHECK(cudaGetLastError());
}
//逐元素乘法
void Hadamard(const GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2, GPUBatchMatrix& out) {
	if (!IsMatch(bat1, bat2)) {
		throw(std::runtime_error("mismatched batch_matrix couldn't hadamard"));
	}
	if (!out.TryMatchWith(bat1.GetOutRowNums(), bat1.GetOutColNums(), bat1.GetInRowNums(), bat1.GetInColNums())) {
		throw std::runtime_error("Calibration out failed in hadamard");
	}
	size_t Size = bat1.GetTotalSize();
	size_t blockNums = (Size + (1024 * 64) - 1) / (1024 * 64);
	size_t threadNums = 1024;
	PerMulTo << <dim3(blockNums, 1, 1), dim3(threadNums, 1, 1) >> > (bat1.GetDataOnlyRead(), bat2.GetDataOnlyRead(),out.GetData(), Size);
	CUDA_CHECK(cudaGetLastError());
}

void HadamardWith(GPUBatchMatrix& bat1, const GPUBatchMatrix& bat2) {
	if (!IsMatch(bat1, bat2)) {
		throw(std::runtime_error("mismatched batch_matrix couldn't hadamard"));
	}
	size_t Size = bat1.GetTotalSize();
	size_t blockNums = (Size + (1024 * 64) - 1) / (1024 * 64);
	size_t threadNums = 1024;
	PerMul << <dim3(blockNums, 1, 1), dim3(threadNums, 1, 1) >> > (bat1.GetData(), bat2.GetDataOnlyRead(), Size);
	CUDA_CHECK(cudaGetLastError());
}
//GPU上的逐元素映射，使用时输入函数模板
//池化
void MaxPoll(const GPUBatchMatrix& bat, GPUBatchMatrix& poll, size_t* antipollBuffer, size_t pollsize) {
	size_t pollInColNums = bat.GetInColNums() / pollsize;
	size_t pollInRowNums = bat.GetInRowNums() / pollsize;
	if (!poll.TryMatchWith(bat.GetOutRowNums(),bat.GetOutColNums(),pollInRowNums,pollInColNums)) {
		throw(std::runtime_error("Calibration out failed in poll"));
	}
	GPU_MaxPoll<<<dim3(bat.GetOutColNums(),bat.GetOutRowNums(),1),dim3(pollInColNums,pollInRowNums) >> >(
		bat.GetDataOnlyRead(),poll.GetData(),antipollBuffer,bat.GetOutRowNums(),bat.GetOutColNums(),bat.GetInRowNums(),bat.GetInColNums(),pollsize
		);
	CUDA_CHECK(cudaGetLastError());
}
//将poll反池化后乘以f加到bat上
//无法保证内维安全，调用方需要注意
void AntiPoll_Timesf_Addto(const GPUBatchMatrix& poll, const size_t* const antipollBuffer, float f, GPUBatchMatrix& bat) {
	size_t threadNums = poll.GetInSize()/16>1024? poll.GetInSize() / 16 : 1024;
	if (bat.GetOutColNums() != poll.GetOutColNums() || bat.GetOutRowNums() != poll.GetOutRowNums()) {
		throw(std::runtime_error("mismatched bat and poll"));
	}
	GPU_AntiPoll_Timesf_Addto << <dim3(poll.GetOutColNums(), poll.GetOutRowNums(), 1), dim3(threadNums, 1, 1) >> > (
		poll.GetDataOnlyRead(), antipollBuffer, bat.GetData(), f, bat.GetInRowNums(), bat.GetInColNums()
		);
	CUDA_CHECK(cudaGetLastError());
}
//Dilate,在该矩阵的内维的每两个元素之间，插入stride-1个0
void DilateTo(const GPUBatchMatrix& bat, size_t stride, GPUBatchMatrix& out, bool setzero) {
	size_t newInColNums = (bat.GetInColNums()-1) * stride + 1;
	size_t newInRowNums = (bat.GetInRowNums()-1) * stride + 1;
	if (!out.TryMatchWith(bat.GetOutRowNums(),bat.GetOutColNums(),newInRowNums,newInColNums)) {
		throw(std::runtime_error("Calibration out failed in Dilate"));
	}
	size_t threadNums = bat.GetInSize() / 16 > 1024 ? bat.GetInSize() / 16 : 1024;
	GPU_Dilate<<<dim3(bat.GetOutColNums(),bat.GetOutRowNums(),1),dim3(threadNums,1,1) >> >(
		bat.GetDataOnlyRead(), out.GetData(), stride, bat.GetInRowNums(), bat.GetInColNums(), setzero
	);
	CUDA_CHECK(cudaGetLastError());
}
//卷积
void GetConvolutionMatrixMul_GPU(const GPUBatchMatrix& bat, const GPUBatchMatrix& ker, GPUBatchMatrix& out, int padding, int stride) {
	if (bat.GetOutColNums() != ker.GetOutRowNums()) {
		throw std::runtime_error("mismatched matrix in convolution");
	}
	if (ker.GetInColNums() != ker.GetInRowNums()) {
		throw std::runtime_error("the ker must be square matirx");
	}
	const int R = bat.GetOutRowNums();      // bat 的行数（前向时 = 批大小；反传 dK 时 = 层通道数）,即样本数
	const int C_in = bat.GetOutColNums();   // bat 的列数（前向时 = 输入通道数；反传 dK 时 = 批大小）
	const int C_out = ker.GetOutColNums();  // 输出通道数
	const int K = ker.GetInColNums();
	const int KK = K * K;

	//bat经过padding后的内维大小
	const int H = bat.GetInRowNums() + 2 * padding;
	const int W = bat.GetInColNums() + 2 * padding;
	/*out的内维大小*/
	//out的内维行数
	const int outInRowNums = (H - K) / stride + 1;
	//out的内维列数
	const int outInColNums = (W - K) / stride + 1;
	if (!out.TryMatchWith(R,C_out, outInRowNums, outInColNums)) {
		throw std::runtime_error("Calibration out failed in convolution");
	}

	int gridy = R;
	int gridx = (C_out + BCG::ppb - 1) / BCG::ppb;
	int gridz = 1;//表示这8个通道由几个block共同完成(bat内再分块，每个block负责自己的一块
	int blocky = (outInRowNums + BCG::rpt - 1) / BCG::rpt;
	int blockx = (outInColNums + BCG::cpt - 1) / BCG::cpt;

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
	if (blockx * BCG::cpt < outInColNums || blocky * BCG::rpt < outInRowNums) {
		gridz = ((outInColNums + blockx * BCG::cpt - 1) / (blockx * BCG::cpt))
			* ((outInRowNums + blocky * BCG::rpt - 1) / (blocky * BCG::rpt));
	}
	Convolution_GPU_kernal<< <dim3(gridx, gridy, gridz), dim3(blockx, blocky, 1), Real_Shared_Size >> >(
		bat.GetDataOnlyRead(), ker.GetDataOnlyRead(), out.GetData(),
		R, C_out, C_in, bat.GetInRowNums(), bat.GetInColNums(), ker.GetInColNums(),
		padding, stride
	);
	CUDA_CHECK(cudaGetLastError());
}
/**/

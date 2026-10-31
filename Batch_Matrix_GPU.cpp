#include"Batch_Matrix_GPU.h"

namespace BCG = Bat_Convolution_Gpu;
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

GPUBatchMatrix::GPUBatchMatrix(const Batch_Matrix& bat) {
	outRowNums_ = bat.GetRowNum();
	outColNums_ = bat.GetColNum();
	inRowNums_ = bat.GetInternalData().size() == 0 ? 0 : bat.GetInternalData()[0].GetRowNum();
	inColNums_ = bat.GetInternalData().size() == 0 ? 0 : bat.GetInternalData()[0].GetColNum();
	if (BCG::capPinnerIn < GetTotalSize()) {
		if (BCG::pinner_in) { cudaFreeHost(BCG::pinner_in); }
		CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&BCG::pinner_in), GetTotalSize() * sizeof(float)));
		BCG::capPinnerIn = GetTotalSize();
	}
	for (int i = 0;i < bat.GetInternalData().size();i++) {
		PackD2F(bat.GetInternalData()[i].GetInternalData().data(), BCG::pinner_in + i * bat.GetInternalData()[0].GetInternalData().size(), bat.GetInternalData()[i].GetInternalData().size());
	}
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), GetTotalSize()* sizeof(float)));
	CUDA_CHECK(cudaMemcpy(data_, BCG::pinner_in, GetTotalSize() * sizeof(float), cudaMemcpyHostToDevice));
}
GPUBatchMatrix::GPUBatchMatrix(int outRowNums, int outColNums, int inRowNums, int inColNums) {
	outRowNums_ = outRowNums;
	outColNums_ = outColNums;
	inRowNums_ = inRowNums;
	inColNums_ = inColNums;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), GetTotalSize() * sizeof(float)));
}
GPUBatchMatrix::GPUBatchMatrix(GPUBatchMatrix&& bat) {
	std::swap(data_, bat.data_);
	outRowNums_ = bat.outRowNums_;
	outColNums_ = bat.outColNums_;
	inRowNums_ = bat.inRowNums_;
	inColNums_ = bat.inColNums_;

}
GPUBatchMatrix::~GPUBatchMatrix() {
	CUDA_CHECK(cudaFree(data_));
}
Batch_Matrix GPUBatchMatrix::ToCPUBat() {
	Batch_Matrix bat(outRowNums_, outColNums_, inRowNums_, inColNums_);
	const size_t outPerBlock = (size_t)inRowNums_ * inColNums_;
	if (BCG::capPinnerOut < GetTotalSize()) {
		if (BCG::pinner_out) { cudaFreeHost(BCG::pinner_out); }
		CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&BCG::pinner_out), GetTotalSize() * sizeof(float)));
		BCG::capPinnerOut = GetTotalSize();
	}
	CUDA_CHECK(cudaMemcpy(BCG::pinner_out, data_, GetTotalSize() * sizeof(float), cudaMemcpyDeviceToHost));
#pragma omp parallel for if(outRowNums_ * outColNums_ > 64)
	for (int i = 0; i < outRowNums_ * outColNums_; ++i) {
		PackF2D(BCG::pinner_out + (size_t)i * outPerBlock,
			Internal::Matrix_Double_Access::GetInterData(Internal::Batch_Matrix_Access::GetInterData(bat)[i]).data(), outPerBlock);
	}
	return bat;
}

bool GPUBatchMatrix::TryMatchWith(size_t aimOutRow, size_t aimOutCol, size_t aimInRow, size_t aimInCol) {
	bool IsMatch =
		aimInCol == inColNums_ &&
		aimInRow == inRowNums_ &&
		aimOutCol == outColNums_ &&
		aimOutRow == outRowNums_;
	if (IsMatch) {
		return true;
	}
	if (GetTotalSize() != aimOutCol * aimInRow * aimInCol * aimOutRow) {
		try {
			if (data_) { CUDA_CHECK(cudaFree(data_)); data_ = nullptr; }
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), aimOutCol * aimInRow * aimInCol * aimOutRow * sizeof(float)));
		}
		catch (std::runtime_error& e) {
			std::cout << "meet error:" << e.what() << std::endl;
			return false;
		}
	}
	inColNums_ = aimInCol;
	inRowNums_ = aimInRow;
	outColNums_ = aimOutCol;
	outRowNums_ = aimOutRow;
	return true;
}
GPUBatchMatrix& GPUBatchMatrix::operator=(GPUBatchMatrix&& bat) {
	std::swap(data_, bat.data_);
	outRowNums_ = bat.outRowNums_;
	outColNums_ = bat.outColNums_;
	inRowNums_ = bat.inRowNums_;
	inColNums_ = bat.inColNums_;

	return *this;
}
bool IsMatch(const GPUBatchMatrix& bat1,const GPUBatchMatrix& bat2) {
	return
		bat1.GetInColNums() == bat2.GetInColNums() &&
		bat1.GetInRowNums() == bat2.GetInRowNums() &&
		bat1.GetOutColNums() == bat2.GetOutColNums() &&
		bat1.GetOutRowNums() == bat2.GetOutRowNums();
}
#define _CRT_SECURE_NO_WARNINGS
#define NETWORK_TEST 0
#include "Complex.h"
#include "Polynomial.h"
#include "Matrix.h"
#include "MathFunction.h"
#include "APP_MNIST.h"
#include "Bench.h"
#include "Batch_Matrix_GPU.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>


namespace sc = std::chrono ;

static double Function_X(double x) { return x; }

int main() {
	// ---- 微基准入口：NN_BENCH=matmul / conv / all（见 Bench.cpp 顶部注释）----
	if (const char* benchOp = std::getenv("NN_BENCH")) { return RunBench(benchOp); }
	// ---- 加速模式：NN_ACCEL = cpu / gpu / conv / matmul（默认 gpu）----
	const char* env = std::getenv("NN_ACCEL");
	const std::string mode = env ? env : "conv";   // 实测：conv 是最佳默认（matmul 走 GPU 反而更慢，见报告）
	const bool useConvGpu = (mode == "gpu" || mode == "conv");
	const bool useMatMulGpu = (mode == "gpu" || mode == "matmul");   // CPU 版矩阵乘在 128×1024×2000 这类形状上更快
	const size_t kNever = (std::numeric_limits<size_t>::max)();
	SetConvGpuThreshold(useConvGpu ? (size_t)8 * 1024 * 1024 : kNever);
	SetMatMulGpuThreshold(useMatMulGpu ? (size_t)2 * 1024 * 1024 : kNever);   // A 路径后 GPU 矩阵乘已反超 CPU（见 bench）
	std::cout << "[bench] NN_ACCEL=" << mode
		<< "  conv=" << (useConvGpu ? "auto(GPU)" : "CPU")
		<< "  matmul=" << (useMatMulGpu ? "auto(GPU)" : "CPU") << std::endl;
#if NETWORK_TEST
	// ---- 网络结构（与原来一致）----
	std::vector<int> kers = { 9, 5, 5 };
	std::vector<int> cnn_ves = { 16, 16, 16 };
	std::vector<ActivationType> cnn_acts = { ActivationType::ReLU, ActivationType::ReLU, ActivationType::Sigmoid };
	std::vector<int> mlp_ves = { 1024, 128, 10 };
	std::vector<ActivationType> mlp_acts = { ActivationType::ReLU, ActivationType::ReLU, ActivationType::None, ActivationType::None };
	auto cnn = std::make_shared<NeuralNetwork_MULCNN>(cnn_ves, kers, cnn_acts);
	cnn->SetPadding({ 6, 2, 2 });
	cnn->SetPoll({ 2, 2, 1 });
	auto mlp = std::make_shared<NeuralNetwork_MLP>(mlp_ves, mlp_acts);
	Complex_NeuralNetwork net({ cnn, mlp });

	// ---- 数据（需在仓库根目录运行）----
	Matrix_Double Feature = LoadMNISTImages("train-images.idx3-ubyte");
	Matrix_Double Result = LabelsToOneHot(LoadMNISTLabels("train-labels.idx1-ubyte"));
	Matrix_Double TextF = LoadMNISTImages("t10k-images.idx3-ubyte");
	Matrix_Double TextR = LabelsToOneHot(LoadMNISTLabels("t10k-labels.idx1-ubyte"));
	std::cout << "[bench] train=" << Feature.GetColNum() << "  test=" << TextF.GetColNum() << std::endl;

	std::mt19937 gen(12345);
	std::uniform_int_distribution<int> distTrain(0, 59999);
	std::uniform_int_distribution<int> distTest(0, 9999);

	const char* envBatch = std::getenv("NN_BATCH");
	const char* envIters = std::getenv("NN_ITERS");
	const char* envEpochs = std::getenv("NN_EPOCHS");
	const int batch = envBatch ? std::atoi(envBatch) : 2000;
	const int itersPerEpoch = envIters ? std::atoi(envIters) : 10;
	const int epochs = envEpochs ? std::atoi(envEpochs) : 10;
	double lr = 0.3;

	// 预热（不进计时）：让 GPU 侧缓冲/锁页内存、内核首次加载都就绪
	{
		std::vector<int> idx(200);
		for (int i = 0; i < 200; ++i) idx[i] = distTrain(gen);
		std::vector<Matrix_Double> x = UnFlattenToBatch(Feature.GetMinorByVec(idx), 28, 28);
		std::vector<Matrix_Double> y = UnFlattenToBatch(Result.GetMinorByVec(idx), 10, 1);
		std::vector<Matrix_Double> z = net.Forward(x);
		net.Back(lr, CrossEntropyGradient(z, y), -9, Function_X);
	}

	for (int ep = 0; ep < epochs; ++ep) {
		auto t0 = std::chrono::steady_clock::now();
		for (int it = 0; it < itersPerEpoch; ++it) {
			std::vector<int> idx(batch);
			for (int i = 0; i < batch; ++i) idx[i] = distTrain(gen);
			std::vector<Matrix_Double> x = UnFlattenToBatch(Feature.GetMinorByVec(idx), 28, 28);
			std::vector<Matrix_Double> y = UnFlattenToBatch(Result.GetMinorByVec(idx), 10, 1);
			std::vector<Matrix_Double> z = net.Forward(x);
			net.Back(lr, CrossEntropyGradient(z, y), -9, Function_X);
			std::cout << (1 + it) * 10 << "%";
		}
		std::cout << std::endl;
		auto t1 = std::chrono::steady_clock::now();
		const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		lr /= 1.03;

		// ---- 测试集准确率（2000 张）----
		const int N = 2000;
		std::vector<int> tidx(N);
		for (int i = 0; i < N; ++i) tidx[i] = distTest(gen);
		std::vector<Matrix_Double> TF = UnFlattenToBatch(TextF.GetMinorByVec(tidx), 28, 28);
		std::vector<Matrix_Double> TR = UnFlattenToBatch(TextR.GetMinorByVec(tidx), 10, 1);
		Matrix_Double logits = GetFlattenMatrixFromBatch(net.Forward(TF));
		Matrix_Double probs = Softmax(logits);
		int hit = 0;
		for (int s = 0; s < N; ++s) {
			int argmax = 0;
			double best = probs.GetData(0, s);
			for (int k = 1; k < 10; ++k) {
				const double p = probs.GetData(k, s);
				if (p > best) { best = p; argmax = k; }
			}
			int label = 0;
			for (int k = 0; k < 10; ++k) if (TR[s].GetData(k, 0) > 0.5) { label = k; break; }
			if (argmax == label) ++hit;
		}
		std::cout << "[bench] epoch " << ep + 1 << "  time=" << (long long)ms<< " ms  acc=" << (double)hit / N << std::endl;
	}
#endif // NETWORK_TEST

	//单纯用于测试GPU的运行速度
	Batch_Matrix bat1(1, 1, 4, 4);
	Batch_Matrix bat2(1, 1, 1, 1);
	bat1.Random(0, 100);
	bat2.Random(0, 100);
	size_t* antipoll = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&antipoll),100*sizeof(size_t)));
	GPUBatchMatrix GPUbat1(bat1);
	GPUBatchMatrix GPUbat2(bat1);
	GPUBatchMatrix GPUbat3;
	MaxPoll(GPUbat1, GPUbat3, antipoll, 2);
	std::vector<AntiPoll_Data> antipoll2=bat1.MaxPoll(2);

	Batch_Matrix bat1_ = GPUbat3.ToCPUBat();
	std::cout << bat1_.GetInternalData()[0] << std::endl;
	std::cout << bat1.GetInternalData()[0] << std::endl;

	AntiPoll_Timesf_Addto(GPUbat3, antipoll, 1, GPUbat1);
	GPUbat1 -= GPUbat2;
	bat1_ = GPUbat1.ToCPUBat();
	std::cout << bat1_.GetInternalData()[0] << std::endl;
	bat2 = GetAntiMaxPoll(bat1, 2, antipoll2);
	std::cout << bat2.GetInternalData()[0] << std::endl;

	Batch_Matrix ker(1, 1, 1, 1);
	Batch_Matrix out(1, 1, 6, 6);
	double test = 0;
	const int N1 = 1;
	auto start1 = sc::steady_clock::now();
	for (int i = 0;i < N1;i++) {
		GetConvolutionMatrixMul_GPU(bat1, ker, out);
		test += out.GetInternalData()[0].GetData(0, 0);
	}
	auto end1 = sc::steady_clock::now();
	auto ms1 = sc::duration_cast<sc::milliseconds>( end1- start1 );
	std::cout << "GPU end，using time:" << ms1 << ",N:" << N1 << std::endl;
	const int N2 = 0;
	auto start2 = sc::steady_clock::now();
	for (int i = 0;i < N2;i++) {
		out=GetConvolutionMatrixMul(bat1,ker);
		test += out.GetInternalData()[0].GetData(0, 0);
	}
	auto end2 = sc::steady_clock::now();
	auto ms2 = sc::duration_cast<sc::milliseconds>( end2 - start2 );
	std::cout << "GPU end，using time:" << ms2 << ",N:" << N2 << std::endl;
	return 0;
}
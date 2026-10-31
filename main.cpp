#define _CRT_SECURE_NO_WARNINGS

// 是否启用神经网络训练（默认开启；没有数据时会自动降级为微基准）
#define NETWORK_TEST 1

#include "Complex.h"
#include "Polynomial.h"
#include "Matrix.h"
#include "MathFunction.h"
#include "APP_MNIST.h"
#include "Bench.h"
#include "Loss.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

static double Function_X(double x) { return x; }

// 依次尝试若干数据目录，返回第一个存在的数据集路径前缀；都不存在返回空串。
static std::string FindDataPrefix() {
	const char* candidates[] = {
		"data/train-images.idx3-ubyte",
		"train-images.idx3-ubyte",
		"Train_data/Fasion/train-images-idx3-ubyte",
		"Train_data/Fashion/train-images-idx3-ubyte",
	};
	for (const char* probe : candidates) {
		std::FILE* f = std::fopen(probe, "rb");
		if (f != nullptr) {
			std::fclose(f);
			const std::string s = probe;
			const std::size_t cut = s.find("train-images");
			if (cut != std::string::npos) return s.substr(0, cut);
		}
	}
	return std::string();
}

int main() {
	// ---- 微基准入口：NN_BENCH=matmul / conv / all（见 Bench.cpp 顶部注释）----
	if (const char* benchOp = std::getenv("NN_BENCH")) { return RunBench(benchOp); }

	// ---- 加速模式：NN_ACCEL = cpu / gpu / conv / matmul（默认 conv）----
	const char* env = std::getenv("NN_ACCEL");
	const std::string mode = env ? env : "conv";
	const bool useConvGpu = (mode == "gpu" || mode == "conv");
	const bool useMatMulGpu = (mode == "gpu" || mode == "matmul");
	const size_t kNever = (std::numeric_limits<size_t>::max)();
	SetConvGpuThreshold(useConvGpu ? (size_t)8 * 1024 * 1024 : kNever);
	SetMatMulGpuThreshold(useMatMulGpu ? (size_t)2 * 1024 * 1024 : kNever);
	std::cout << "[main] NN_ACCEL=" << mode
		<< "  conv=" << (useConvGpu ? "auto(GPU)" : "CPU")
		<< "  matmul=" << (useMatMulGpu ? "auto(GPU)" : "CPU") << std::endl;

#if NETWORK_TEST
	const std::string prefix = FindDataPrefix();
	if (prefix.empty()) {
		std::cout << "[main] 未找到 MNIST/Fashion-MNIST 数据（应为 data/train-images.idx3-ubyte 等 4 个文件）。\n"
			<< "[main] 数据已随仓库提供（data/ 目录）；若缺失请见 README『数据集与许可』一节。\n"
			<< "[main] 现在改为运行微基准（不需要数据集）..." << std::endl;
		return RunBench("all");
	}
	std::cout << "[main] 数据目录: " << prefix << std::endl;

	// ---- 网络结构（MNIST 28x28：3 层卷积 + 2 层全连接）----
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

	// ---- 数据 ----
	Matrix_Double Feature = LoadMNISTImages(prefix + "train-images.idx3-ubyte");
	Matrix_Double Result = LabelsToOneHot(LoadMNISTLabels(prefix + "train-labels.idx1-ubyte"));
	Matrix_Double TextF = LoadMNISTImages(prefix + "t10k-images.idx3-ubyte");
	Matrix_Double TextR = LabelsToOneHot(LoadMNISTLabels(prefix + "t10k-labels.idx1-ubyte"));
	std::cout << "[main] train=" << Feature.GetColNum() << "  test=" << TextF.GetColNum() << std::endl;

	std::mt19937 gen(12345);
	std::uniform_int_distribution<int> distTrain(0, 59999);
	std::uniform_int_distribution<int> distTest(0, 9999);

	const char* envBatch = std::getenv("NN_BATCH");
	const char* envIters = std::getenv("NN_ITERS");
	const char* envEpochs = std::getenv("NN_EPOCHS");
	const int batch = envBatch ? std::atoi(envBatch) : 2000;
	const int itersPerEpoch = envIters ? std::atoi(envIters) : 10;
	const int epochs = envEpochs ? std::atoi(envEpochs) : 2;   // 默认短跑，便于快速验证
	double lr = 0.3;

	// 预热（不计时）：让 GPU 缓冲/锁页内存/内核首次加载就绪
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
		}
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
		std::cout << "[train] epoch " << ep + 1 << "  time=" << (long long)ms
			<< " ms  acc=" << (double)hit / N << std::endl;
	}
#endif // NETWORK_TEST
	return 0;
}

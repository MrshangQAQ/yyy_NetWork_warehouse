// =====================================================================================
// 高等代数 —— CPU 演示版（无需 CUDA / 无需 SFML / 无需数据集）
//
// 目的：让没有 NVIDIA 显卡和 CUDA 环境的同学也能直接编译运行，快速了解这个项目
//       "从零实现"了哪些东西、跑出来大概是什么水平。
//
// 内容：
//   1. 数学库：精确有理数运算、矩阵乘法基准
//   2. MLP：从零实现的多层感知机，在合成 XOR 数据上训练
//   3. CNN：从零实现的多通道卷积网络，在合成的 8×8 条形图分类任务上训练
//
// 构建：打开 Demo_CPU.vcxproj → Release | x64 → 生成并运行（F5）
// =====================================================================================

#define _CRT_SECURE_NO_WARNINGS
#ifndef CPU_ONLY
#define CPU_ONLY
#endif

#include "Matrix.h"
#include "Batch_Matrix.h"
#include "NeuralNetwork.h"
#include "Loss.h"
#include "Rational.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>   // SetConsoleOutputCP：让中文输出在 CMD / VS 控制台正常显示
#endif

namespace {

	using Clock = std::chrono::steady_clock;

	double Ms(Clock::duration d) {
		return std::chrono::duration<double, std::milli>(d).count();
	}

	void PrintHeader() {
		std::cout << "==============================================================\n";
		std::cout << " 高等代数 —— 从零实现的 C++ 数学库 / 神经网络（CPU 演示版）\n";
		std::cout << " 无需 CUDA / SFML / 数据集；仅需 Visual Studio 与 C++20\n";
		std::cout << "==============================================================\n";
		std::cout << "[环境] 硬件线程数 = " << std::thread::hardware_concurrency() << "\n\n";
	}

	// ============================== 1. 数学库 ==============================
	void DemoMath() {
		std::cout << "===== 1. 数学库 =====" << std::endl;

		// 精确有理数：1/2 + 1/3 + 1/6 应当精确等于 1（浮点会得到 0.999...）
		const Rational half(1, 2), third(1, 3), sixth(1, 6);
		const Rational sum = half + third + sixth;
		std::cout << "[Rational] 1/2 + 1/3 + 1/6 = "
			<< sum.num << "/" << sum.den
			<< "，精确等于 1 ? " << ((sum == Rational(1, 1)) ? "是" : "否")
			<< "（double 版本 = " << (0.5 + 1.0 / 3.0 + 1.0 / 6.0) << "）" << std::endl;

		// 矩阵乘法基准：Matrix_Double 为行主序连续存储 + OpenMP + AVX2 + 寄存器分块
		std::cout << "[Matrix] CPU 矩阵乘法基准（单次）：" << std::endl;
		const int sizes[] = { 256, 512, 1024 };
		for (int n : sizes) {
			Matrix_Double A(n, n), B(n, n);
			A.Random(0.0, 1.0);
			B.Random(0.0, 1.0);
			Matrix_Double C = A * B;   // 预热一次（含首次分配与缓存建立）
			double bestMs = 1e300;
			for (int rep = 0; rep < 3; ++rep) {
				const auto t0 = Clock::now();
				C = A * B;
				const auto t1 = Clock::now();
				bestMs = (std::min)(bestMs, Ms(t1 - t0));
			}
			const double gflops = 2.0 * (double)n * n * n / (bestMs / 1000.0) / 1e9;
			std::cout << "  " << n << "^3 : " << bestMs << " ms（3 次最优）, "
				<< gflops << " GFLOP/s（校验值 C[0][0]=" << C.GetData(0, 0) << "）" << std::endl;
		}
		std::cout << std::endl;
	}

	// ============================== 评估工具 ==============================
	double EvalAccuracy(Complex_NeuralNetwork& net,
		const Matrix_Double& Feature, const Matrix_Double& Result,
		int labelRows, int h, int w) {
		const int N = Feature.GetColNum();
		std::vector<int> idx(N);
		std::iota(idx.begin(), idx.end(), 0);
		std::vector<Matrix_Double> x = UnFlattenToBatch(Feature.GetMinorByVec(idx), h, w);
		std::vector<Matrix_Double> y = UnFlattenToBatch(Result.GetMinorByVec(idx), labelRows, 1);
		Matrix_Double probs = Softmax(GetFlattenMatrixFromBatch(net.Forward(x)));
		int hit = 0;
		for (int s = 0; s < N; ++s) {
			int argmax = 0;
			double best = probs.GetData(0, s);
			for (int c = 1; c < labelRows; ++c) {
				const double p = probs.GetData(c, s);
				if (p > best) { best = p; argmax = c; }
			}
			for (int c = 0; c < labelRows; ++c) {
				if (y[s].GetData(c, 0) > 0.5) {
					if (argmax == c) ++hit;
					break;
				}
			}
		}
		return (double)hit / (double)N;
	}

	// ============================== 2. MLP ==============================
	double DemoMLP() {
		std::cout << "===== 2. MLP（从零实现，合成 XOR 数据）=====" << std::endl;

		const int N = 800, F = 2, C = 2;
		std::mt19937 gen(2026);
		std::uniform_real_distribution<double> U(-1.0, 1.0);
		std::vector<double> feat((size_t)F * N), labs((size_t)C * N, 0.0);
		for (int i = 0; i < N; ++i) {
			const double x = U(gen), y = U(gen);
			feat[(size_t)0 * N + i] = x;
			feat[(size_t)1 * N + i] = y;
			const int label = (x * y > 0.0) ? 1 : 0;   // 象限 XOR
			labs[(size_t)label * N + i] = 1.0;
		}
		Matrix_Double Feature(std::move(feat), F, N);
		Matrix_Double Result(std::move(labs), C, N);

		auto mlp = std::make_shared<NeuralNetwork_MLP>(
			std::vector<int>{ F, 32, C },
			std::vector<ActivationType>{ ActivationType::None, ActivationType::ReLU, ActivationType::None });
		Complex_NeuralNetwork net({ mlp });

		double lastAcc = 0.0;
		const int batch = 100, itersPerEpoch = 10, epochs = 15;
		std::mt19937 idxGen(7);
		std::uniform_int_distribution<int> dist(0, N - 1);
		double lr = 0.5;

		for (int ep = 0; ep < epochs; ++ep) {
			const auto t0 = Clock::now();
			for (int it = 0; it < itersPerEpoch; ++it) {
				std::vector<int> idx(batch);
				for (int& v : idx) v = dist(idxGen);
				auto x = UnFlattenToBatch(Feature.GetMinorByVec(idx), F, 1);
				auto y = UnFlattenToBatch(Result.GetMinorByVec(idx), C, 1);
				auto logits = net.Forward(x);
				net.Back(lr, CrossEntropyGradient(logits, y), 0.0, nullptr);
			}
			lr *= 0.95;
			const auto t1 = Clock::now();
			const double acc = EvalAccuracy(net, Feature, Result, C, F, 1);
			lastAcc = acc;
			std::cout << "  epoch " << (ep + 1) << "  训练耗时 " << (long long)Ms(t1 - t0)
				<< " ms  准确率 " << acc << std::endl;
		}
		std::cout << std::endl;
		return lastAcc;
	}


	// ============================== 3. CNN ==============================
	double DemoCNN() {
		std::cout << "===== 3. CNN（从零实现，合成 8x8 条形图分类）=====" << std::endl;

		const int H = 8, W = 8, N = 600, C = 2;
		std::mt19937 gen(42);
		std::uniform_real_distribution<double> U(0.0, 1.0);
		std::vector<double> feat((size_t)H * W * N, 0.0);
		std::vector<double> labs((size_t)C * N, 0.0);

		for (int i = 0; i < N; ++i) {
			const int label = i % 2;                 // 0 = 横条，1 = 竖条
			const int shift = (int)(gen() % 3) - 1;  // 随机平移，避免记住固定位置
			for (int r = 0; r < H; ++r) {
				for (int c = 0; c < W; ++c) {
					double v = (U(gen) < 0.15) ? 1.0 : 0.0;   // 15% 噪声
					if (label == 0) {
						if (r == 3 + shift || r == 4 + shift) v = 1.0;
					}
					else {
						if (c == 3 + shift || c == 4 + shift) v = 1.0;
					}
					feat[((size_t)r * W + c) * N + i] = v;
				}
			}
			labs[(size_t)label * N + i] = 1.0;
		}
		Matrix_Double Feature(std::move(feat), H * W, N);
		Matrix_Double Result(std::move(labs), C, N);

		// 8x8 输入 → conv(3x3, p=1) → pool(2) → 4x4 → conv(3x3, p=1) → pool(1) → 4x4
		// 最后一层 4 通道 × 4×4 = 64 维特征，接 MLP {64, 16, 2}
		auto cnn = std::make_shared<NeuralNetwork_MULCNN>(
			std::vector<int>{ 4, 4 },
			std::vector<int>{ 3, 3 },
			std::vector<ActivationType>{ ActivationType::ReLU, ActivationType::ReLU });
		cnn->SetPadding({ 1, 1 });
		cnn->SetPoll({ 2, 1 });
		auto mlp = std::make_shared<NeuralNetwork_MLP>(
			std::vector<int>{ 64, 16, 2 },
			std::vector<ActivationType>{ ActivationType::None, ActivationType::ReLU, ActivationType::None });
		Complex_NeuralNetwork net({ cnn, mlp });

		double lastAcc = 0.0;
		const int batch = 100, itersPerEpoch = 8, epochs = 12;
		std::mt19937 idxGen(11);
		std::uniform_int_distribution<int> dist(0, N - 1);
		double lr = 0.5;

		for (int ep = 0; ep < epochs; ++ep) {
			const auto t0 = Clock::now();
			for (int it = 0; it < itersPerEpoch; ++it) {
				std::vector<int> idx(batch);
				for (int& v : idx) v = dist(idxGen);
				auto x = UnFlattenToBatch(Feature.GetMinorByVec(idx), H, W);
				auto y = UnFlattenToBatch(Result.GetMinorByVec(idx), C, 1);
				auto logits = net.Forward(x);
				net.Back(lr, CrossEntropyGradient(logits, y), 0.0, nullptr);
			}
			lr *= 0.95;
			const auto t1 = Clock::now();
			const double acc = EvalAccuracy(net, Feature, Result, C, H, W);
			lastAcc = acc;
			std::cout << "  epoch " << (ep + 1) << "  训练耗时 " << (long long)Ms(t1 - t0)
				<< " ms  准确率 " << acc << std::endl;
		}
		std::cout << std::endl;
		return lastAcc;
	}

} // namespace

int main() {
	const auto tAll0 = Clock::now();
	PrintHeader();

	DemoMath();
	const double mlpAcc = DemoMLP();
	const double cnnAcc = DemoCNN();

	const auto tAll1 = Clock::now();
	std::cout << "===== 总计 =====" << std::endl;
	std::cout << "MLP 最终准确率（合成 XOR）: " << mlpAcc << std::endl;
	std::cout << "CNN 最终准确率（8x8 条形图，15% 噪声）: " << cnnAcc << std::endl;
	std::cout << "演示总耗时 " << (long long)Ms(tAll1 - tAll0) << " ms" << std::endl;
	std::cout << "说明：以上全部为 CPU 实现；完整工程支持 CUDA 加速（见 README）。" << std::endl;
	return 0;
}

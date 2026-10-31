#define _CRT_SECURE_NO_WARNINGS
#include "Bench.h"
#include "Matrix.h"
#include "Batch_Matrix.h"
#if !defined(CPU_ONLY)
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// =====================================================================================
// 单算子微基准
//
// 用法（在仓库根目录运行）：
//   NN_BENCH=matmul  NN_BENCH_ITERS=20 .\x64\Release\高等代数.exe
//   NN_BENCH=conv    NN_BENCH_ITERS=6  .\x64\Release\高等代数.exe
//   NN_BENCH=all     .\run_bench.ps1
//
// 可调环境变量：
//   NN_BENCH_SHAPE  矩阵乘形状 "MxKxN"（默认 128x1024x2000）
//   NN_BENCH_MODES  逗号分隔：矩阵乘 cpu,legacy,a；卷积 cpu,gpu
//   NN_BENCH_ITERS  采样次数（默认 20 / 卷积 6）
//   NN_BENCH_WARMUP 预热次数（默认 5 / 卷积 2）
//
// 采样方式：不同模式在同一进程内交替跑，机器漂移/温度对每种模式影响相同；
// 报告同一模式的 min / median / mean，并换算有效 GFLOP/s。
// =====================================================================================
namespace {
	using SteadyClock = std::chrono::steady_clock;

	double Ms(SteadyClock::duration d) {
		return std::chrono::duration<double, std::milli>(d).count();
	}

	int EnvInt(const char* name, int def) {
		const char* p = std::getenv(name);
		return (p != nullptr) ? std::atoi(p) : def;
	}

	struct Stat { double minMs = 0.0, medMs = 0.0, meanMs = 0.0; };

	Stat Summarize(std::vector<double> v) {
		Stat s;
		if (v.empty()) return s;
		std::sort(v.begin(), v.end());
		s.minMs = v.front();
		s.medMs = v[v.size() / 2];
		double sum = 0.0;
		for (double x : v) sum += x;
		s.meanMs = sum / (double)v.size();
		return s;
	}

	// 防止编译器把"结果没被使用"的调用优化掉
	volatile double gSink = 0.0;

	// 是否真的有可用的 CUDA 设备（无 GPU / 驱动不匹配 / 被环境变量强制关闭时返回 false）
	bool HasCudaDevice() {
#if defined(CPU_ONLY)
		return false;
#else
		if (std::getenv("NN_BENCH_NO_GPU") != nullptr) return false;
		int count = 0;
		if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return false;
		return true;
#endif
	}

	// 去掉当前机器跑不了的 GPU 模式
	void FilterCudaModes(std::vector<std::string>& modes) {
		if (HasCudaDevice()) return;
		const bool wasGpu = std::find(modes.begin(), modes.end(), "gpu") != modes.end()
			|| std::find(modes.begin(), modes.end(), "a") != modes.end()
			|| std::find(modes.begin(), modes.end(), "legacy") != modes.end();
		modes.erase(std::remove_if(modes.begin(), modes.end(), [](const std::string& m) {
			return m == "gpu" || m == "a" || m == "legacy";
			}), modes.end());
		if (wasGpu) std::cout << "[bench] no usable CUDA device: testing CPU only (GPU modes skipped)" << std::endl;
	}
	std::vector<std::string> SplitModes(const char* text, std::vector<std::string> def) {
		if (text == nullptr) return def;
		std::vector<std::string> out;
		std::string cur;
		for (const char* p = text; *p != '\0'; ++p) {
			if (*p == ',') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
			else cur.push_back(*p);
		}
		if (!cur.empty()) out.push_back(cur);
		return out.empty() ? def : out;
	}

	// ============================== 矩阵乘 ==============================
	double RunMatMulOnce(const std::string& mode, const Matrix_Double& mat1, const Matrix_Double& mat2, Matrix_Double& out) {
		// 只有 GPU 的两种路径关心这个开关
		SetMatMulTransferMode(mode == "legacy" ? 0 : 1);
		const auto t0 = SteadyClock::now();
		if (mode == "cpu") {
			Matrix_Double tmp = mat1 * mat2;   // CPU 路径同样走"返回后赋值"，计入分配
			out = std::move(tmp);
		}
		else {
			Matrix_Double_Mutiple_GPU(mat1, mat2, out);
		}
		const auto t1 = SteadyClock::now();
		gSink += out.GetData(0, 0);
		return Ms(t1 - t0);
	}

	void BenchMatMul() {
		int M = 128, K = 1024, N = 2000;
		if (const char* s = std::getenv("NN_BENCH_SHAPE")) {
			std::sscanf(s, "%dx%dx%d", &M, &K, &N);
		}
		const int iters = EnvInt("NN_BENCH_ITERS", 20);
		const int warmup = EnvInt("NN_BENCH_WARMUP", 5);
		std::vector<std::string> modes = SplitModes(std::getenv("NN_BENCH_MODES"), { "cpu", "legacy", "a" });
		FilterCudaModes(modes);

		Matrix_Double mat1(M, K), mat2(K, N);
		mat1.Random(0.0, 1.0);
		mat2.Random(0.0, 1.0);
		Matrix_Double out;

		std::cout << "[bench] matmul " << M << "x" << K << "x" << N
			<< "  iters=" << iters << "  warmup=" << warmup << std::endl;

		for (const auto& mode : modes) {
			for (int i = 0; i < warmup; ++i) RunMatMulOnce(mode, mat1, mat2, out);
		}
		std::vector<std::vector<double>> samples(modes.size());
		for (int it = 0; it < iters; ++it) {
			for (size_t mi = 0; mi < modes.size(); ++mi) {
				samples[mi].push_back(RunMatMulOnce(modes[mi], mat1, mat2, out));
			}
		}

		const double gflops = 2.0 * (double)M * K * N / 1e9;
		std::printf("%-9s %10s %12s %12s %12s\n", "mode", "min(ms)", "median(ms)", "mean(ms)", "GFLOP/s");
		for (size_t mi = 0; mi < modes.size(); ++mi) {
			const Stat s = Summarize(samples[mi]);
			std::printf("%-9s %10.3f %12.3f %12.3f %12.1f\n",
				modes[mi].c_str(), s.minMs, s.medMs, s.meanMs, gflops / (s.minMs / 1000.0));
		}
		std::printf("(GFLOP/s computed from min(ms))\n");
	}

	// ============================== 卷积 ==============================
	struct ConvShape { int R, Cin, Cout, H, W, K, padding, stride; const char* name; };

	double RunConvOnce(const ConvShape& sh, const Batch_Matrix& bat, const Batch_Matrix& ker, Batch_Matrix& out, bool useGpu) {
		const size_t kNever = ((size_t)1 << 62);
		SetConvGpuThreshold(useGpu ? 0 : kNever);
		const auto t0 = SteadyClock::now();
		GetConvolutionMatrixMul_Auto(bat, ker, out, sh.padding, sh.stride);
		const auto t1 = SteadyClock::now();
		if (!out.GetInternalData().empty()) gSink += out.GetInternalData()[0].GetData(0, 0);
		return Ms(t1 - t0);
	}

	void BenchConv() {
		// 与 MNIST CNN 的配置一致的前向形状（batch=2000）
		const ConvShape shapes[] = {
			{ 2000, 1,  8, 32, 32, 9, 6, 2, "fwd0" },
			{ 2000, 8,  8, 16, 16, 5, 2, 2, "fwd1" },
			{ 2000, 8, 16,  8,  8, 5, 2, 1, "fwd2" },
		};
		const int iters = EnvInt("NN_BENCH_ITERS", 6);
		const int warmup = EnvInt("NN_BENCH_WARMUP", 2);
		std::vector<std::string> modes = SplitModes(std::getenv("NN_BENCH_MODES"), { "cpu", "gpu" });
		FilterCudaModes(modes);

		for (const auto& sh : shapes) {
			Batch_Matrix bat(sh.R, sh.Cin, sh.H, sh.W);
			Batch_Matrix ker(sh.Cin, sh.Cout, sh.K, sh.K);
			Batch_Matrix out;
			// 先跑一次 GPU，让 out 得到实际输出尺寸（也完成锁页/显存缓冲分配）
			RunConvOnce(sh, bat, ker, out, true);
			const int ho = out.GetInternalData()[0].GetRowNum();
			const int wo = out.GetInternalData()[0].GetColNum();
			const double macs = (double)sh.R * ho * wo * sh.Cin * sh.K * sh.K * sh.Cout;
			const double gflops = 2.0 * macs / 1e9;
			std::cout << "[bench] conv " << sh.name
				<< "  R=" << sh.R << " Cin=" << sh.Cin << " Cout=" << sh.Cout
				<< " K=" << sh.K << " p=" << sh.padding << " s=" << sh.stride
				<< "  out=" << ho << "x" << wo
				<< "  macs=" << macs / 1e6 << "M" << std::endl;

			std::vector<std::vector<double>> samples(modes.size());
			for (int it = 0; it < iters; ++it) {
				for (size_t mi = 0; mi < modes.size(); ++mi) {
					samples[mi].push_back(RunConvOnce(sh, bat, ker, out, modes[mi] == "gpu"));
				}
			}
			std::printf("%-9s %10s %12s %12s %12s\n", "mode", "min(ms)", "median(ms)", "mean(ms)", "GFLOP/s");
			for (size_t mi = 0; mi < modes.size(); ++mi) {
				const Stat s = Summarize(samples[mi]);
				std::printf("%-9s %10.3f %12.3f %12.3f %12.1f\n",
					modes[mi].c_str(), s.minMs, s.medMs, s.meanMs, gflops / (s.minMs / 1000.0));
			}
		}
	}
} // namespace

int RunBench(const char* op) {
	const std::string what = (op != nullptr) ? op : "all";
	if (what == "matmul") { BenchMatMul(); return 0; }
	if (what == "conv") { BenchConv(); return 0; }
	BenchMatMul();
	BenchConv();
	return 0;
}

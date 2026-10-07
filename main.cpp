#include"Complex.h"
#include "Polynomial.h"
#include "Matrix.h"
#include "MathFunction.h"
#include "APP_Generate_Function_Graph.h"
#include "APP_MNIST.h"
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <Windows.h>
#include <omp.h>

static Polynomial x({ Complex(0,0),Complex(1,0) });
static Polynomial I(Complex(1,0));

/// <summary>【临时自检】im2col 对拍，函数实现在 Batch_Matrix.cpp。
/// 验证完成后：删掉本行声明，以及 main() 开头的那次调用即可。</summary>
bool SelfTest_GetConvolutionMatrixMul();

static double Function_X(double x) {
	return x;
}
int main() {

	std::cout << "threads = " << omp_get_max_threads() << std::endl;
	// 【临时自检】把新的 im2col 卷积实现与朴素实现逐元素对拍，确认数值一致
	/*
	SelfTest_GetConvolutionMatrixMul();

	std::vector<int>kers1 = {9,5,5};
	std::vector<int>cnn_ves = {16,16,16};
	std::vector<ActivationType>cnn_acts1 = { ActivationType::ReLU,ActivationType::ReLU,ActivationType::Sigmoid };
	std::vector<int> mlp_ves1 = { 1024,64,10};
	std::vector<ActivationType>mlp_acts1 = {ActivationType::ReLU,ActivationType::ReLU,ActivationType::None,ActivationType::None};
	auto cnn1 = std::make_shared<NeuralNetwork_MULCNN>(cnn_ves, kers1, cnn_acts1);
	cnn1->SetPadding({ 6,2,2 });          // 注意：shared_ptr 用 -> 访问成员
	//28*28->32*32->16*16->16*16->8*8->8*8->4*4
	cnn1->SetPoll({2,2,1});
	auto mlp1 = std::make_shared<NeuralNetwork_MLP>(mlp_ves1, mlp_acts1);
	Complex_NeuralNetwork ner({cnn1,mlp1});
	Matrix_Double Feature =LoadMNISTImages("train-images.idx3-ubyte");
	std::cout << "Train featur load:" << Feature.GetColNum() << "," << Feature.GetRowNum() << std::endl;

	Matrix_Double Result = LabelsToOneHot(LoadMNISTLabels("train-labels.idx1-ubyte"));
	std::cout << "Train Result load:"<< Result.GetColNum() << "," << Result.GetRowNum() << std::endl;

	Matrix_Double Text_Feature = LoadMNISTImages("t10k-images.idx3-ubyte");
	std::cout << "Text featur load:" << Text_Feature.GetColNum() << "," << Text_Feature.GetRowNum() << std::endl;

	Matrix_Double Text_Result= LabelsToOneHot(LoadMNISTLabels("t10k-labels.idx1-ubyte"));
	std::cout << "Text Result load:" << Text_Result.GetColNum() << "," << Text_Result.GetRowNum() << std::endl;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, 59999);//获取随机数
	std::uniform_int_distribution<int> dist2(0, 9999);
	double oldacc = -1;
	double meanProb = 0;
	double steplenth = 0.3;
	for (int k = 0;k < 30;k++) {
		auto start = std::chrono::steady_clock::now();
		if (k < 15) {
			for (int j = 0;j < 100;j++) {
				std::vector<int> index(200, 0);
				for (int i = 0;i < 200;i++) {
					index[i] = dist(gen);
				}
				std::vector<Matrix_Double> Train_Feature = UnFlattenToBatch(Feature.GetMinorByVec(index), 28, 28);
				std::vector<Matrix_Double> Train_Result = UnFlattenToBatch(Result.GetMinorByVec(index), 10, 1);
				std::vector<Matrix_Double> z = ner.Forward(Train_Feature);
				ner.Back(steplenth, CrossEntropyGradient(z, Train_Result), -9, Function_X);
				std::cout << j + 1 << "% ";
			}//先进行10次，少量高频数据训练
		}
		else {
			for (int j = 0;j < 10;j++) {
				std::vector<int> index(2000, 0);
				for (int i = 0;i < 2000;i++) {
					index[i] = dist(gen);
				}
				std::vector<Matrix_Double> Train_Feature = UnFlattenToBatch(Feature.GetMinorByVec(index), 28, 28);
				std::vector<Matrix_Double> Train_Result = UnFlattenToBatch(Result.GetMinorByVec(index), 10, 1);
				std::vector<Matrix_Double> z = ner.Forward(Train_Feature);
				ner.Back(steplenth, CrossEntropyGradient(z, Train_Result), -9, Function_X);
				std::cout << (j + 1) * 10 << "% ";
			}//再进行10次，大量低频数据训练
		}
		auto end = std::chrono::steady_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << std::endl;
		std::cout << "using time:" << ms << std::endl;
		steplenth /= 1.03;
		const int N = 1000;                       // 用完整测试集，消除抽样噪声
		std::vector<int> allIndex(N);
		for (int i = 0; i < N; i++) allIndex[i] = dist2(gen);

		std::vector<Matrix_Double> TF = UnFlattenToBatch(Text_Feature.GetMinorByVec(allIndex), 28, 28);
		std::vector<Matrix_Double> TR = UnFlattenToBatch(Text_Result.GetMinorByVec(allIndex), 10, 1);

		Matrix_Double logits = GetFlattenMatrixFromBatch(ner.Forward(TF));   // 10 × N，每列一个样本
		Matrix_Double probs = Softmax(logits);                              // 沿列（跨行）归一化

		int hit = 0, miss = 0;
		double sumAll = 0.0, sumHit = 0.0, sumMiss = 0.0;
		for (int s = 0; s < N; s++) {
			int argmax = 0;                          // 概率最大的类别
			double best = probs.GetData(0, s);
			for (int k = 1; k < 10; k++) {
				double p = probs.GetData(k, s);
				if (p > best) { best = p; argmax = k; }
			}
			int label = 0;                           // one-hot 中为 1 的那一位就是真实标签
			for (int k = 0; k < 10; k++) {
				if (TR[s].GetData(k, 0) > 0.5) { label = k; break; }
			}
			double pCorrect = probs.GetData(label, s);
			sumAll += pCorrect;
			if (argmax == label) { hit++;  sumHit += pCorrect; }   // 答对
			else { miss++; sumMiss += pCorrect; }   // 答错
		}
		std::cout << "Train of " << k + 1 << " over data:" << std::endl;
		std::cout << "argmax accuracy           = " << (double)hit / N << std::endl;
		std::cout << "mean p(correct) [now]     = " << sumAll / N << std::endl;
		std::cout << "mean p(correct | correct) = " << sumHit / hit << std::endl;   // = c_hit
		std::cout << "mean p(correct | error)   = " << sumMiss / miss << std::endl;   // = c_miss
	}
	APP_MNIST Window(ner);
	while (Window.IsRunning()) {
		Window.ProcessEvent();
		Window.Render();
	}
	*/
	std::vector<double> mat1_data = {
		1,2,3,
		4,5,6,
		7,8,9
	};
	std::vector<double> mat2_data = { 
		1,2,3,
		4,5,6,
		7,8,9
	};
	const int N = 100;
	const int Mat_Size = 4096;
	const int Mat_Size2 = 2047;
	const int row = 4093;
	const int col = 672;
	Matrix_Double mat1(Mat_Size);
	mat1.Random(0, 500);
	Matrix_Double mat2(Mat_Size);
	mat2.Random(0, 500);
	double sink = 0;
	Matrix_Double r(Mat_Size, Mat_Size);   //结果缓冲：循环外分配一次，之后每轮复用（零分配、零清零）
	auto stard = std::chrono::steady_clock::now();
	for (int i = 0; i < N; i++) {
		Matrix_Double_Mutiple_GPU(mat1, mat2, r);
		sink += r.GetData(0, 0);        // 只为防止调用被优化掉
	}
	auto end = std::chrono::steady_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - stard);
	std::cout<<"ave:" << ms/N  << std::endl;
	Matrix_Double_Mutiple_GPU(mat1, mat2, r);
	//Matrix_Double tmp = mat1 * mat2;
	//std::cout << tmp.GetData(row, col) << "," << r.GetData(row, col);
}

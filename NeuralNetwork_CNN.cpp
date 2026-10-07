#include"NeuralNetwork.h"

static std::vector<Matrix_Double> BackPropagation_CNN(const Layer_CNN& layer) {
	std::vector<Matrix_Double> copy(layer.a.size());
	if (layer.act == ActivationType::None) {
		for (int i = 0;i < copy.size();i++) {
			std::vector < double > one_matrix_data;
			one_matrix_data.resize(layer.a[i].GetRowNum() * layer.a[i].GetColNum(), 1);
			copy[i] = Matrix_Double(one_matrix_data, layer.a[i].GetRowNum(), layer.a[i].GetColNum());
		}
		return copy;
	}
	switch (layer.act)
	{
	case ActivationType::Sigmoid:
		for (int i=0;i < copy.size();i++) {
			copy[i] = SigmoidGradient(layer.a[i]);
		}
		return copy;
	case ActivationType::ReLU:
		for (int i=0;i < copy.size();i++) {
			copy[i] = ReLUGradient(layer.z[i]);
		}
		return copy;
	case ActivationType::Tanh:
		for (int i=0;i < copy.size();i++) {
			copy[i] = TanhGradient(layer.a[i]);
		}
		return copy;
	default:
		break;
	}
	throw std::runtime_error("Unknow action type");
}//反向传播，得到导数

NeuralNetwork_CNN::NeuralNetwork_CNN(std::vector<int>Ker_Sizes, std::vector<ActivationType>acts_) {
	if (acts_.size() < Ker_Sizes.size()) {
		acts_.resize(Ker_Sizes.size(), ActivationType::ReLU);
		acts_.back() = ActivationType::None;
	}
	Layers_.resize(Ker_Sizes.size());
	for (int i = 0;i < Layers_.size();i++) {
		Layers_[i] = Layer_CNN{
			Matrix_Double(Ker_Sizes[i]),
			acts_[i],
			0,
			1,
			{},
			{}
		};
	}
	for (int i = 0;i < Layers_.size();i++) {
		double fan_in = (double)(Ker_Sizes[i]*Ker_Sizes[i]);
		double w_std = std::sqrt(1.0 / fan_in);
		Layers_[i].Ker.Random(0, w_std);
	}
}
File_Handle_Inf NeuralNetwork_CNN::Load(std::fstream& file, int beginIn) {
	file.clear();
	file.seekg(beginIn);
	//第一步，读出该节点的层数信息和类型
	int layers_nums =0;
	file.read(reinterpret_cast<char*>(&type), sizeof(NeuralNodeType));
	if (type != NeuralNodeType::CNN) {
		type = NeuralNodeType::CNN;
		throw std::runtime_error("mismatched file of the network");
	}
	file.read(reinterpret_cast<char*>(&layers_nums), sizeof(int));
	//第二步，读出该节点的每层的相关数据
	std::vector<int> ves(layers_nums);
	std::vector<int> paddings(layers_nums);
	std::vector<int> strides(layers_nums);
	std::vector<ActivationType> acts(layers_nums);
	for (int i = 0;i < layers_nums;i++) {
		file.read(reinterpret_cast<char*>(&ves[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&paddings[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&strides[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&acts[i]), sizeof(ActivationType));
	}
	//第三步，读出该节点的每层的具体数据
	for (int i = 0;i < layers_nums;i++) {
		std::vector<double> mat(ves[i]*ves[i]);
		file.read(reinterpret_cast<char*>(&mat[0]), mat.size() * sizeof(double));
		Layers_[i] = {
			Matrix_Double(mat,ves[i],ves[i]),
			acts[i],
			paddings[i],
			strides[i],
			{},
			{}
		};
	}
	int p = file.tellp();
	return { file.good(),p };
}
File_Handle_Inf NeuralNetwork_CNN::Save(std::fstream& file, int beginIn) {
	file.clear();
	file.seekp(beginIn);
	//第一步，存入该节点的类型与该节点的层数
	int layers_nums = Layers_.size();
	file.write(reinterpret_cast<const char*>(&type), sizeof(NeuralNodeType));
	file.write(reinterpret_cast<const char*>(&layers_nums), sizeof(int));
	//第二步，存入该节点各层的大小和其余数据
	for (int i = 0;i < layers_nums;i++) {
		int ve = Layers_[i].Ker.GetColNum();
		file.write(reinterpret_cast<const char*>(&ve), sizeof(int));//这里因为是方阵，所以我们只存储边长
		file.write(reinterpret_cast<const char*>(&Layers_[i].p), sizeof(int));
		file.write(reinterpret_cast<const char*>(&Layers_[i].s), sizeof(int));
		file.write(reinterpret_cast<const char*>(&Layers_[i].act), sizeof(ActivationType));
	}
	//第三步，存入该节点的各个矩阵
	for (int i = 0;i < layers_nums;i++) {
		const std::vector<double>& mat = Layers_[i].Ker.GetInternalData();
		file.write(reinterpret_cast<const char*>(mat.data()), mat.size() * sizeof(double));
	}
	int g = file.tellp();
	return { file.good(),g };
}
void NeuralNetwork_CNN::SetPadding(std::vector<int>paddings) {
	int size = paddings.size() < Layers_.size() ? paddings.size() : Layers_.size();
	for (int i = 0;i < size;i++) {
		Layers_[i].p = paddings[i];
		if (Layers_[i].p < 0) {
			throw std::runtime_error("negative paddings");
		}
	}
}
void NeuralNetwork_CNN::SetStride(std::vector<int>strides) {
	int size = strides.size() < Layers_.size() ? strides.size() : Layers_.size();
	for (int i = 0;i < size;i++) {
		Layers_[i].s = strides[i];
		if (Layers_[i].s < 1) {
			throw std::runtime_error("negative or zero strides");
		}
	}
}    
std::vector<Matrix_Double> NeuralNetwork_CNN::Forward(const std::vector<Matrix_Double>& DataIn) {
	InputBuffer = DataIn;
	Layers_[0].z = Convolution(DataIn, Layers_[0].Ker, Layers_[0].p, Layers_[0].s);
	Layers_[0].a = Activate(Layers_[0].z, Layers_[0].act);
	for (int i = 1;i < Layers_.size();i++) {

		Layers_[i].z = Convolution(Layers_[i-1].a, Layers_[i].Ker,Layers_[i].p,Layers_[i].s);
		Layers_[i].a = Activate(Layers_[i].z, Layers_[i].act);
	}
	return Layers_.back().a;
}
std::vector<Matrix_Double> NeuralNetwork_CNN::Back(double step_Length, const std::vector<Matrix_Double>& LossGradient, double lambda, double (*regular)(double)) {
	std::vector<Matrix_Double> dA = LossGradient;//损失对于本层输出的梯度，即希望本层往哪个方向改变
	int BackSize = dA.size();
	step_Length /= (double)BackSize;
	for (int i = Layers_.size() - 1;i >= 1;i--) {
		std::vector<Matrix_Double> delta = Hadamard(dA, BackPropagation_CNN(Layers_[i]));

		Matrix_Double dK(Layers_[i].Ker.GetRowNum(), Layers_[i].Ker.GetColNum());
		Matrix_Double K_Rotato180 = Layers_[i].Ker;
		K_Rotato180.Rotato_180();
		if (regular != nullptr) {
			dK += std::exp(lambda) * MapMatrix(Layers_[i].Ker, regular);
		}
		for (int j = 0;j < BackSize;j++) {
			dK += Convolution(GetPaddingMatrix(Layers_[i - 1].a[j], Layers_[i].p), GetDilateMatrix(delta[j], Layers_[i].s));
			dA[j] = Convolution(GetPaddingMatrix(GetDilateMatrix(delta[j], Layers_[i].s), Layers_[i].Ker.GetColNum() - 1 - Layers_[i].p), K_Rotato180);
		}
		Layers_[i].Ker -= step_Length * dK;
	}
	std::vector<Matrix_Double> delta = Hadamard(dA, BackPropagation_CNN(Layers_[0]));

	Matrix_Double dK(Layers_[0].Ker.GetRowNum(), Layers_[0].Ker.GetColNum());
	Matrix_Double K_Rotato180 = Layers_[0].Ker;
	K_Rotato180.Rotato_180();
	for (int j = 0;j < BackSize;j++) {
		dK += Convolution(GetPaddingMatrix(InputBuffer[j], Layers_[0].p), GetDilateMatrix(delta[j], Layers_[0].s));
		dA[j] = Convolution(GetPaddingMatrix(GetDilateMatrix(delta[j], Layers_[0].s), Layers_[0].Ker.GetColNum() - 1 - Layers_[0].p), K_Rotato180);
	}
	Layers_[0].Ker -= step_Length * dK;
	return dA;
}
#include"NeuralNetwork.h"


static Matrix_Double BackPropagation_MLP(const Layer_MLP& layer) {
	if (layer.act == ActivationType::None) {
		std::vector < double > one_matrix_data;
		one_matrix_data.resize(layer.a.GetRowNum() * layer.a.GetColNum(), 1);
		return Matrix_Double(one_matrix_data, layer.a.GetRowNum(), layer.a.GetColNum());
	}
	switch (layer.act)
	{
	case ActivationType::Sigmoid:
		return SigmoidGradient(layer.a);
	case ActivationType::ReLU:
		return ReLUGradient(layer.z);
	case ActivationType::Tanh:
		return TanhGradient(layer.a);
	default:
		break;
	}
	throw std::runtime_error("Unknow action type");
}//反向传播，得到导数

NeuralNetwork_MLP::NeuralNetwork_MLP(std::vector<int>layers_ves, std::vector<ActivationType>acts_) {
	if (acts_.size() < layers_ves.size()) {
		acts_.resize(layers_ves.size(), ActivationType::ReLU);
		acts_.back() = ActivationType::None;
	}
	Layers_.resize(layers_ves.size());
	Layers_[0] = Layer_MLP{
			Matrix_Double(layers_ves[0],1),
			Matrix_Double(layers_ves[0],1),
			ActivationType::None,
			Matrix_Double(layers_ves[0],1),
			Matrix_Double(layers_ves[0],1)
	};//最外层，相当于存储输入的数据
	for (int i = 1;i < Layers_.size();i++) {
		Layers_[i] = Layer_MLP{
			Matrix_Double(layers_ves[i],layers_ves[i - 1]),
			Matrix_Double(layers_ves[i],1),
			acts_[i],
			Matrix_Double(layers_ves[i],1),
			Matrix_Double(layers_ves[i],1)
		};
	}
	for (int i = 1;i < Layers_.size();i++) {
		double fan_in = (double)layers_ves[i - 1];   
		double w_std = std::sqrt(1.0 / fan_in);     
		Layers_[i].W.Random(0, w_std);
		Layers_[i].b.Random(0, 0.01);
	}
};
File_Handle_Inf NeuralNetwork_MLP::Save(std::fstream& file,int beginIn) {
	file.clear();
	file.seekp(beginIn);
	//第一步，先写入该神经网络的层数
	int layers_size = Layers_.size();//每一层的输出，其中不用存储第一层的输入，因为第一层的输入就等于第一层的输出
	file.write(reinterpret_cast<const char*>(&type), sizeof(NeuralNodeType));
	file.write(reinterpret_cast<const char*>(&layers_size),sizeof(int));
	//第二步，再写入各层的输出维数和对应的act所占的空间大小,第0层也存，相当于存储第一层的输入维数
	for (int i = 0;i < Layers_.size();i++) {
		int ve = Layers_[i].W.GetRowNum();
		file.write(reinterpret_cast<const char*>(&ve), sizeof(int));
		file.write(reinterpret_cast<const char*>(&Layers_[i].act), sizeof(ActivationType));
	}
	//第三步，依次存入W和b,第0层不存，读取的最后时再单独将其单位化
	for (int i = 1;i < Layers_.size();i++) {
		// 必须取 vector 内部元素数组的首地址（.data()）；写成 &GetInternalData() 取到的是 vector 对象自身的地址，
		// 会写出垃圾数据，并且从该地址连续读 size*8 字节属于越界读取，可能直接崩溃
		const std::vector<double>& W_data = Layers_[i].W.GetInternalData();
		const std::vector<double>& b_data = Layers_[i].b.GetInternalData();
		file.write(reinterpret_cast<const char*>(W_data.data()), W_data.size() * sizeof(double));
		file.write(reinterpret_cast<const char*>(b_data.data()), b_data.size() * sizeof(double));
	}
	int p = file.tellp();
	return {file.good(),p};
}
File_Handle_Inf NeuralNetwork_MLP::Load(std::fstream& file, int beginIn) {
	file.clear();
	file.seekg(beginIn);
	//第一步，读入该神经网络的层数
	int layers_size = 0;
	file.read(reinterpret_cast<char*>(&type), sizeof(NeuralNodeType));
	if (type != NeuralNodeType::MLP) {
		type = NeuralNodeType::MLP;
		throw std::runtime_error("mismatched file of the network");
	}
	file.read(reinterpret_cast<char*>(&layers_size), sizeof(int));
	if (layers_size <= 0) {
		throw std::runtime_error("error number of layers");
	}
	// 必须先校验再 resize：若 layers_size 为负数，resize 会先把它转成巨大的 size_t 而抛出 bad_alloc
	Layers_.resize(layers_size);
	//第二步，读入该神经网络每层的基本信息
	std::vector<int>layer_ves(layers_size,0);
	std::vector<ActivationType>acts(layers_size,ActivationType::None);
	for (int i = 0;i < layers_size; i++) {
		file.read(reinterpret_cast<char*>(&layer_ves[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&acts[i]), sizeof(ActivationType));
	}
	//第三步，读入该神经网络的矩阵数据,这里没有读入第0层的，第0层单独处理
	for (int i = 1;i < layers_size;i++) {
		std::vector < double > W_data(layer_ves[i - 1] * layer_ves[i], 0.0);
		std::vector < double > b_data(layer_ves[i], 0.0);
		file.read(reinterpret_cast<char*>(&W_data[0]),W_data.size()*sizeof(double));
		file.read(reinterpret_cast<char*>(&b_data[0]),b_data.size()*sizeof(double));
		Layers_[i].W = Matrix_Double(W_data, layer_ves[i], layer_ves[i-1]);
		Layers_[i].b = Matrix_Double(b_data, layer_ves[i], 1);
		Layers_[i].act = acts[i];
	}
	//最后单独设置0层矩阵
	Layers_[0].W = Matrix_Double(layer_ves[0]);
	Layers_[0].b = Matrix_Double(layer_ves[0], 1);
	Layers_[0].act = acts[0];
	int g = file.tellg();
	return { file.good(),g };
}
Matrix_Double NeuralNetwork_MLP::Forward(const Matrix_Double& DataIn) {
	Matrix_Double predict = DataIn;
	Layers_[0].a = DataIn;
	for (int i = 1;i < Layers_.size();i++) {
		Layers_[i].z = AddBias(Layers_[i].W * predict, Layers_[i].b);
		Layers_[i].a = Activate(Layers_[i].z, Layers_[i].act);
		predict = Layers_[i].a;
	}
	return predict;
}
Matrix_Double NeuralNetwork_MLP::Back(double step_Length, const Matrix_Double& LossGradient, double lambda, double (*regular)(double)) {
	Matrix_Double dA = LossGradient;//对本层的输出的梯度(即下一层希望本层的输出往哪个方向走)
	for (int i = Layers_.size()-1;i >= 1;i--) {//第0层是存储输入数据的地方，反向传播时跳过
		Matrix_Double delta = Hadamard(dA, BackPropagation_MLP(Layers_[i]));
		double invBatch = 1.0 / (double)delta.GetColNum();

		Matrix_Double dW = invBatch*delta * GetTranslateMatrix(Layers_[i - 1].a);
		Matrix_Double db = invBatch*RowSums(delta);

		dA = GetTranslateMatrix(Layers_[i].W) * delta;//dA现在变为，本层希望上一层的输出往哪个方向走

		if (regular != nullptr) {
			dW += std::exp(lambda) * MapMatrix(Layers_[i].W, regular);
		}

		Layers_[i].W -= step_Length*dW;
		Layers_[i].b -= step_Length*db;
	}
	return dA;
}    
std::vector<Matrix_Double> NeuralNetwork_MLP::Forward(const std::vector<Matrix_Double>& batch) {
	batch_col_buffer = batch[0].GetColNum();
	batch_row_buffer = batch[0].GetRowNum();
	if (batch_row_buffer*batch_col_buffer != Layers_[0].W.GetRowNum()) {
		std::cout << batch_row_buffer * batch_col_buffer << std::endl;
		throw std::runtime_error("the batch couldn't use to the network");
	}
	Matrix_Double flatten = GetFlattenMatrixFromBatch(batch);
	Matrix_Double result = Forward(flatten);
	std::vector<Matrix_Double> result_batch = UnFlattenToBatch(result,Layers_.back().W.GetRowNum(),1);
	return result_batch;
}
std::vector<Matrix_Double> NeuralNetwork_MLP::Back(double lr, const std::vector<Matrix_Double>& grad, double lambda, double (*regular)(double)){
	int w = grad.size();
	if (grad[0].GetColNum() * grad[0].GetRowNum() != Layers_.back().W.GetRowNum()) {
		throw std::runtime_error("the batch couldn't use to the network");
	}
	Matrix_Double flatten = GetFlattenMatrixFromBatch(grad);
	Matrix_Double result = Back(lr,flatten,lambda,regular);
	std::vector<Matrix_Double> result_batch = UnFlattenToBatch(result, batch_row_buffer, batch_col_buffer);
	return result_batch;
}
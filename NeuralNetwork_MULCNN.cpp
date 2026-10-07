#include"NeuralNetwork.h"

static Batch_Matrix BackPropagation_MULCNN(const Layer_MULCNN& layer) {
	if (layer.act == ActivationType::None) {
		if (layer.a.GetInternalData().size() == 0) {
			throw std::runtime_error("empty input");
		}
		int OuterSize = layer.a.GetInternalData().size();
		int OuterRowNums = layer.a.GetRowNum();
		int OuterColNums = layer.a.GetColNum();

		int InterSize = layer.a.GetInternalData()[0].GetInternalData().size();
		int InterRowNums = layer.a.GetInternalData()[0].GetRowNum();
		int InterColNums = layer.a.GetInternalData()[0].GetColNum();
		std::vector<Matrix_Double> NewData(OuterSize);
		for (int i = 0;i < NewData.size();i++) {
			std::vector < double > one_matrix_data;
			one_matrix_data.resize(InterSize, 1);
			NewData[i] = Matrix_Double(std::move(one_matrix_data), InterRowNums,InterColNums);
		}
		return Batch_Matrix(std::move(NewData),OuterRowNums,OuterColNums);
	}
	switch (layer.act)
	{
	case ActivationType::Sigmoid:
		return MapMatrix(layer.a,SigmoidGradientScalar);
	case ActivationType::ReLU:
		return MapMatrix(layer.a,ReLUGradientScalar);//这里特殊，a是池化后的z，故用a也成立
	case ActivationType::Tanh:
		return MapMatrix(layer.a,TanhGradientScalar);
	default:
		break;
	}
	throw std::runtime_error("Unknow action type");
}//反向传播，得到导数

NeuralNetwork_MULCNN::NeuralNetwork_MULCNN(std::vector<int>Layers_ves, std::vector<int> Ker_Sizes, std::vector<ActivationType> acts_) {
	int size = Layers_ves.size() < Ker_Sizes.size() ? Layers_ves.size() : Ker_Sizes.size();

	if (acts_.size() < size) {
		acts_.resize(size, ActivationType::ReLU);
		acts_.back() = ActivationType::None;
	}
	//这里统一一下，前向传播时，Batch矩阵的每一行表示了一个输入矩阵的全部特征
	Layers_.resize(size);
	Layers_[0] = {
		Batch_Matrix(1,Layers_ves[0],Ker_Sizes[0],Ker_Sizes[0]),
		acts_[0],
		0,
		1,
		1,
		Batch_Matrix(),
		Batch_Matrix(),
		{}
	};
	for (int i = 1;i < size;i++) {
		Layers_[i] = {
		Batch_Matrix(Layers_ves[i - 1],Layers_ves[i],Ker_Sizes[i],Ker_Sizes[i]),
		acts_[i],
		0,
		1,
		1,
		Batch_Matrix(),
		Batch_Matrix(),
		{}
		};
	}
	for (int i = 0;i < size;i++) {
		double fan_in = (double)(Ker_Sizes[i] * Ker_Sizes[i]);
		double w_std = std::sqrt(1.0 / fan_in);
		Layers_[i].Kers.Random(0,w_std);
	}
}
void NeuralNetwork_MULCNN::SetPadding(std::vector<int>paddings) {
	int size = paddings.size() < Layers_.size() ? paddings.size() : Layers_.size();
	for (int i = 0;i < size;i++) {
		Layers_[i].p = paddings[i];
		if (Layers_[i].p < 0) {
			throw std::runtime_error("negative paddings");
		}
	}
}
void NeuralNetwork_MULCNN::SetStride(std::vector<int>strides) {
	int size = strides.size() < Layers_.size() ? strides.size() : Layers_.size();
	for (int i = 0;i < size;i++) {
		Layers_[i].s = strides[i];
		if (Layers_[i].s < 1) {
			throw std::runtime_error("negative or zero strides");
		}
	}
}
void NeuralNetwork_MULCNN::SetPoll(std::vector<int>polls) {
	int size = polls.size() < Layers_.size() ? polls.size() : Layers_.size();
	for (int i = 0;i < size;i++) {
		Layers_[i].poll = polls[i];
		if (Layers_[i].poll < 1) {
			throw std::runtime_error("negative or zero strides");
		}
	}
}
std::vector<Matrix_Double> NeuralNetwork_MULCNN::Forward(const std::vector<Matrix_Double>& DataIn) {
	InputBuffer = Batch_Matrix( DataIn,DataIn.size(),1);
	Layers_[0].z = GetConvolutionMatrixMul(InputBuffer, Layers_[0].Kers, Layers_[0].p, Layers_[0].s);
	if (Layers_[0].act == ActivationType::None ||
		Layers_[0].act == ActivationType::ReLU ||
		Layers_[0].act == ActivationType::Sigmoid ||
		Layers_[0].act == ActivationType::Tanh) {
		Layers_[0].a = Layers_[0].z;
		//对于单调函数，可以先池化，再激活,节省性能
		Layers_[0].antipoll_buffer = Layers_[0].a.MaxPoll(Layers_[0].poll);
		Layers_[0].a = Activate(Layers_[0].a, Layers_[0].act);
	}
	else {
		Layers_[0].a = Activate(Layers_[0].z, Layers_[0].act);
		Layers_[0].antipoll_buffer = Layers_[0].a.MaxPoll(Layers_[0].poll);
	}
	for (int i = 1;i < Layers_.size();i++) {
		Layers_[i].z = GetConvolutionMatrixMul(Layers_[i - 1].a, Layers_[i].Kers, Layers_[i].p, Layers_[i].s);
		if (Layers_[i].act == ActivationType::None || 
			Layers_[i].act == ActivationType::ReLU || 
			Layers_[i].act == ActivationType::Sigmoid || 
			Layers_[i].act == ActivationType::Tanh) {
			Layers_[i].a = Layers_[i].z;
			//对于单调函数，可以先池化，再激活,节省性能
			Layers_[i].antipoll_buffer = Layers_[i].a.MaxPoll(Layers_[i].poll);
			Layers_[i].a = Activate(Layers_[i].a, Layers_[i].act);
		}
		else {
			Layers_[i].a = Activate(Layers_[i].z, Layers_[i].act);
			Layers_[i].antipoll_buffer = Layers_[i].a.MaxPoll(Layers_[i].poll);
		}
	}
	//重新整理a中最后的数据，把他的每一行都压成一个矩阵,保存原来a中矩阵的列数
	outputColNum_Buffer = Layers_.back().a.GetInternalData()[0].GetColNum();
	return FuseKernelMaps(Layers_.back().a);
}
std::vector<Matrix_Double> NeuralNetwork_MULCNN::Back(double step_Length, const std::vector<Matrix_Double>& LossGradient, double lambda, double (*regular)(double)) {
	Batch_Matrix dA = SplitKernelGrad(LossGradient, outputColNum_Buffer);
	int BackSize = LossGradient.size();
	step_Length /= (double)BackSize;
	for (int i = (int)Layers_.size() - 1;i >= 1;i--) {
		Batch_Matrix dZ= GetAntiMaxPoll(Hadamard(dA, BackPropagation_MULCNN(Layers_[i])), Layers_[i].poll, Layers_[i].antipoll_buffer);
		/*
		Batch_Matrix AP = GetBlockTranspose( GetPaddingMatrix(Layers_[i-1].a, Layers_[i].p));
		Batch_Matrix ZD = GetDilateMatrix(dZ, Layers_[i].s);
		Batch_Matrix ZDP = GetPaddingMatrix(ZD, Layers_[i].Kers.GetInternalData()[0].GetColNum()-1-Layers_[i].p);
		*/
		Batch_Matrix ZD = GetDilateMatrix(dZ, Layers_[i].s);
		Batch_Matrix K_Rotato180 = GetBlockTranspose(Layers_[i].Kers);//先转置在内部取反和先内部取反在转置应该没差别
		K_Rotato180.InterRotato_180();
		//dA=Con(ZDP,K_Rorato180)
		dA = GetConvolutionMatrixMul(GetPaddingMatrix(ZD, Layers_[i].Kers.GetInternalData()[0].GetColNum() - 1 - Layers_[i].p), K_Rotato180);
		//dK=Con(AP,ZD
		Batch_Matrix dK = GetConvolutionMatrixMul(GetBlockTranspose(GetPaddingMatrix(Layers_[i - 1].a, Layers_[i].p)), ZD);
		if (regular != nullptr) {
			dK += BackSize* std::exp(lambda) * MapMatrix(Layers_[i].Kers, regular);
		}
		Layers_[i].Kers-= step_Length*dK;
	}

	Batch_Matrix dZ = GetAntiMaxPoll(Hadamard(dA, BackPropagation_MULCNN(Layers_[0])), Layers_[0].poll, Layers_[0].antipoll_buffer);
	Batch_Matrix ZD = GetDilateMatrix(dZ, Layers_[0].s);
	Batch_Matrix K_Rotato180 = GetBlockTranspose(Layers_[0].Kers);
	K_Rotato180.InterRotato_180();
	dA = GetConvolutionMatrixMul(GetPaddingMatrix(ZD, Layers_[0].Kers.GetInternalData()[0].GetColNum() - 1 - Layers_[0].p), K_Rotato180);
	Batch_Matrix dK = GetConvolutionMatrixMul(GetBlockTranspose(GetPaddingMatrix(InputBuffer, Layers_[0].p)), ZD);
	if (regular != nullptr) {
		dK += std::exp(lambda) * MapMatrix(Layers_[0].Kers, regular);
	}
	Layers_[0].Kers -= step_Length * dK;
	return  dA.GetInternalData();
}
File_Handle_Inf NeuralNetwork_MULCNN::Load(std::fstream& file, int beginIn) {
	file.clear();
	file.seekg(beginIn);
	//第一步，读入该节点的类型与层数
	int layers_nums = 0;
	file.read(reinterpret_cast<char*>(&type), sizeof(NeuralNodeType));
	if (type != NeuralNodeType::MULCNN) {
		type = NeuralNodeType::MULCNN;
		throw std::runtime_error("mismatched file of the network");
	}
	file.read(reinterpret_cast<char*>(&layers_nums), sizeof(int));
	//先校验再 resize：若为负数，resize 会把它转成巨大的 size_t 抛 bad_alloc
	if (layers_nums <= 0) {
		throw std::runtime_error("error number of layers");
	}
	Layers_.resize(layers_nums);
	//第二步，读入每一层的结构信息
	std::vector<int> ker_row_nums(layers_nums, 0);
	std::vector<int> ker_col_nums(layers_nums, 0);
	std::vector<int> ker_sizes(layers_nums, 0);
	std::vector<int> paddings(layers_nums, 0);
	std::vector<int> strides(layers_nums, 1);
	std::vector<ActivationType> acts(layers_nums, ActivationType::None);
	std::vector<int> polls(layers_nums, 1);
	for (int i = 0;i < layers_nums;i++) {
		file.read(reinterpret_cast<char*>(&ker_row_nums[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&ker_col_nums[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&ker_sizes[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&paddings[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&strides[i]), sizeof(int));
		file.read(reinterpret_cast<char*>(&acts[i]), sizeof(ActivationType));
		file.read(reinterpret_cast<char*>(&polls[i]), sizeof(int));
	}
	//第三步，逐层读回卷积核数据，重建 Kers 网格
	for (int i = 0;i < layers_nums;i++) {
		int block_nums = ker_row_nums[i] * ker_col_nums[i];
		int block_size = ker_sizes[i] * ker_sizes[i];
		std::vector<Matrix_Double> blocks(block_nums, Matrix_Double(ker_sizes[i], ker_sizes[i]));
		for (int j = 0;j < block_nums;j++) {
			std::vector<double> one(block_size, 0.0);
			file.read(reinterpret_cast<char*>(one.data()), block_size * sizeof(double));
			blocks[j] = Matrix_Double(std::move(one), ker_sizes[i], ker_sizes[i]);
		}
		Layers_[i].Kers = Batch_Matrix(std::move(blocks), ker_row_nums[i], ker_col_nums[i]);
		Layers_[i].act = acts[i];
		Layers_[i].p = paddings[i];
		Layers_[i].s = strides[i];
		Layers_[i].poll = polls[i];
	}
	int endIn = (int)file.tellg();
	return { file.good(), endIn };
}
File_Handle_Inf NeuralNetwork_MULCNN::Save(std::fstream& file, int beginIn) {
	file.clear();
	file.seekp(beginIn);
	//第一步，写入该节点的类型与层数
	int layers_nums = (int)Layers_.size();
	file.write(reinterpret_cast<const char*>(&type), sizeof(NeuralNodeType));
	file.write(reinterpret_cast<const char*>(&layers_nums), sizeof(int));
	//第二步，写入每一层的结构信息：核网格行数(输入通道)、列数(输出通道)、核边长，以及 p、s、act
	for (int i = 0;i < layers_nums;i++) {
		int ker_row_nums = Layers_[i].Kers.GetRowNum();                 //网格行数 = 输入通道数
		int ker_col_nums = Layers_[i].Kers.GetColNum();                 //网格列数 = 输出通道数
		int ker_size = Layers_[i].Kers.GetInternalData()[0].GetRowNum();//核是方阵，只存边长
		file.write(reinterpret_cast<const char*>(&ker_row_nums), sizeof(int));
		file.write(reinterpret_cast<const char*>(&ker_col_nums), sizeof(int));
		file.write(reinterpret_cast<const char*>(&ker_size), sizeof(int));
		file.write(reinterpret_cast<const char*>(&Layers_[i].p), sizeof(int));
		file.write(reinterpret_cast<const char*>(&Layers_[i].s), sizeof(int));
		file.write(reinterpret_cast<const char*>(&Layers_[i].act), sizeof(ActivationType));
		file.write(reinterpret_cast<const char*>(&Layers_[i].poll), sizeof(int));
	}
	//第三步，逐层写入所有卷积核，每个核是一个 ker_size×ker_size 的矩阵
	for (int i = 0;i < layers_nums;i++) {
		const std::vector<Matrix_Double>& blocks = Layers_[i].Kers.GetInternalData();
		for (int j = 0;j < (int)blocks.size();j++) {
			// 必须取 vector 内部元素数组的首地址(.data())，写成 &GetInternalData() 会写出垃圾并越界读
			const std::vector<double>& data_xss_removed = blocks[j].GetInternalData();
			file.write(reinterpret_cast<const char*>(data_xss_removed.data()), data_xss_removed.size() * sizeof(double));
		}
	}
	int endIn = (int)file.tellp();
	return { file.good(), endIn };
}

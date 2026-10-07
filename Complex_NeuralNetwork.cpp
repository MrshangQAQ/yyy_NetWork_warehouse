#include"NeuralNetwork.h"

Complex_NeuralNetwork::Complex_NeuralNetwork(std::vector<std::shared_ptr<NeuralNetwork>> Nodes_) :
	NeuralNodes(Nodes_)
{}
std::vector<Matrix_Double> Complex_NeuralNetwork::Forward(const std::vector<Matrix_Double>& DataIn) {
	std::vector<Matrix_Double> forward = DataIn;
	for (int i = 0;i < NeuralNodes.size();i++) {
		forward=NeuralNodes[i]->Forward(forward);
	}
	return forward;
}
std::vector<Matrix_Double> Complex_NeuralNetwork::Back(double step_Length, const std::vector<Matrix_Double>& LossGradient, double lambda, double (*regular)(double)) {
	std::vector<Matrix_Double>dA = LossGradient;
	for (int i = NeuralNodes.size() - 1;i >= 0;i--) {
		dA=NeuralNodes[i]->Back(step_Length,dA,lambda,regular);
	}
	return dA;
}
File_Handle_Inf Complex_NeuralNetwork::Load(std::fstream& file) {
	file.clear();
	file.seekg(0);
	//第一步，读出该复杂神经网络有几个节点
	int NodeNums = 0;
	file.read(reinterpret_cast<char*>(&NodeNums), sizeof(int));
	if (NodeNums < 0) {
		throw std::runtime_error("nagetive NodeNums,please check the file");
	}
	//第二步，读出该复杂每一层节点的索引
	int BaseAddress = 1 + file.tellp();
	file.seekg(BaseAddress);
	std::vector<int>NodeBaseAddress(NodeNums + 1, 0);
	file.read(reinterpret_cast<char*>(&NodeBaseAddress[0]), NodeBaseAddress.size() * sizeof(int));
	//第三步，根据NodeBaseAddress读出每个节点
	for (int i = 0;i < NodeNums;i++) {
		file.seekg(NodeBaseAddress[i]);
		NeuralNodeType type;
		file.read(reinterpret_cast<char*>(&type), sizeof(NeuralNodeType));
		switch (type)
		{
		case NeuralNodeType::MLP: {
			NeuralNetwork_MLP mlp;
			mlp.Load(file, NodeBaseAddress[i]);
			NeuralNodes[i] = std::make_shared<NeuralNetwork_MLP>(mlp);
		}
			break;
		case NeuralNodeType::CNN: {
			NeuralNetwork_CNN cnn(std::vector<int>{1},std::vector<ActivationType>{ActivationType::None});
			cnn.Load(file, NodeBaseAddress[i]);
			NeuralNodes[i] = std::make_shared<NeuralNetwork_CNN>(cnn);
		}
			break;
		case NeuralNodeType::MULCNN: {
			NeuralNetwork_MULCNN cnn;
			cnn.Load(file, NodeBaseAddress[i]);
			NeuralNodes[i] = std::make_shared<NeuralNetwork_MULCNN>(cnn);
		} 
			break;
		default:
			throw std::runtime_error("unknow neural_node type");
			break;
		}
	}
}
File_Handle_Inf Complex_NeuralNetwork::Save(std::fstream& file) {
	int NodeNums = NeuralNodes.size();
	file.clear();
	file.seekp(0);
	std::vector<int>NodeBaseAddress(NodeNums+1,0);
	//第一步，写入节点，并记录此时的file的位置
	file.write(reinterpret_cast<const char*>(&NodeNums), sizeof(int));
	int BaseAddress = 1 + file.tellp();
	file.seekp(BaseAddress);
	//第二步，为后面的节点预留空间
	file.write(reinterpret_cast<const char*>(&NodeBaseAddress[0]), NodeBaseAddress.size() * sizeof(int));
	NodeBaseAddress[0] = 1 + file.tellp();
	//第三步，开始写入每个节点
	for (int i = 0;i < NodeNums;i++) {
		File_Handle_Inf inf=NeuralNodes[i]->Save(file, NodeBaseAddress[i]);
		if (!inf.is_good) {
			throw std::runtime_error("meet error in try to save the complex_neural_network");
		}
		NodeBaseAddress[i + 1] = inf.endIn + 1;
	}
	file.seekp(BaseAddress);
	//第四步，把每个节点的信息会传如文件中
	file.write(reinterpret_cast<const char*>(&NodeBaseAddress[0]), NodeBaseAddress.size() * sizeof(int));
	return { file.good(),NodeBaseAddress.back() };
}

void Complex_NeuralNetwork::AddNode(const std::shared_ptr<NeuralNetwork>& node) {
	if (node == nullptr) {
		throw std::runtime_error("null node in AddNode");
	}
	NeuralNodes.push_back(node);
}

void Complex_NeuralNetwork::InsertNode(int index, const std::shared_ptr<NeuralNetwork>& node) {
	if (node == nullptr) {
		throw std::runtime_error("null node in InsertNode");
	}
	// 插入允许 index == 个数（插到末尾），故上界是 size() 而不是 size()-1
	if (index < 0 || index > (int)NeuralNodes.size()) {
		throw std::runtime_error("index out of range in InsertNode");
	}
	NeuralNodes.insert(NeuralNodes.begin() + index, node);
}

void Complex_NeuralNetwork::DeleteNode(int index) {
	// 删除不允许 index == 个数；容器为空时任何 index 都越界
	if (index < 0 || index >= (int)NeuralNodes.size()) {
		throw std::runtime_error("index out of range in DeleteNode");
	}
	NeuralNodes.erase(NeuralNodes.begin() + index);
}

void Complex_NeuralNetwork::SetNode(int index, const std::shared_ptr<NeuralNetwork>& node) {
	if (node == nullptr) {
		throw std::runtime_error("null node in SetNode");
	}
	if (index < 0 || index >= (int)NeuralNodes.size()) {
		throw std::runtime_error("index out of range in SetNode");
	}
	NeuralNodes[index] = node;
}

void Complex_NeuralNetwork::ClearNodes() {
	NeuralNodes.clear();
}

int Complex_NeuralNetwork::GetNodeNum() const {
	return (int)NeuralNodes.size();
}

std::shared_ptr<NeuralNetwork> Complex_NeuralNetwork::GetNode(int index) const {
	if (index < 0 || index >= (int)NeuralNodes.size()) {
		throw std::runtime_error("index out of range in GetNode");
	}
	return NeuralNodes[index];
}
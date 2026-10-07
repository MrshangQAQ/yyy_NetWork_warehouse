#pragma once
#include"Matrix.h"
#include"Batch_Matrix.h"
#include<fstream>
#include<cstdint>
/// <summary>Sigmoid 标量核：σ(z) = 1/(1+e^-z)</summary>
inline static double SigmoidScalar(double z) { return 1.0 / (1.0 + std::exp(-z)); }

/// <summary>逐元素 Sigmoid：a = σ(z)，值域 (0,1)</summary>
inline Matrix_Double Sigmoid(const Matrix_Double& z) { return MapMatrix(z, SigmoidScalar); }

/// <summary>Sigmoid 导数标量核：σ'(z) = a(1-a)</summary>
inline static double SigmoidGradientScalar(double a) { return a * (1.0 - a); }

/// <summary>逐元素 Sigmoid 导数。</summary>
/// <param name="a">必须是前向已算出的 a = σ(z)，不是激活前的 z（利用 σ'=a(1-a) 省一次 exp）</param>
/// <returns>与 a 同形的导数矩阵</returns>
inline Matrix_Double SigmoidGradient(const Matrix_Double& a) { return MapMatrix(a, SigmoidGradientScalar); }

/// <summary>ReLU 标量核：max(0, z)</summary>
inline static double ReLUScalar(double z) { return z > 0.0 ? z : 0.0; }

/// <summary>逐元素 ReLU</summary>
inline Matrix_Double ReLU(const Matrix_Double& z) { return MapMatrix(z, ReLUScalar); }

/// <summary>ReLU 导数标量核：z>0 取 1，否则取 0（z=0 处不可导，工程上取 0）</summary>
inline static double ReLUGradientScalar(double z) { return z > 0.0 ? 1.0 : 0.0; }

/// <summary>逐元素 ReLU 导数。</summary>
/// <param name="z">必须是激活前的 z（需要 z 的符号来判断），不可传激活后的 a</param>
/// <returns>与 z 同形的导数矩阵</returns>
inline Matrix_Double ReLUGradient(const Matrix_Double& z) { return MapMatrix(z, ReLUGradientScalar); }

/// <summary>按行求和：对每一行把该行所有列元素相加，得到 n×1 的列向量矩阵。</summary>
/// <param name="mat">输入矩阵 n×m（神经网络中约定每列是一个样本）</param>
/// <returns>n×1 矩阵，第 i 个元素为 mat 第 i 行之和</returns>

inline static double TanhScalar(double z) {
    double expz = std::exp(z);
    double RECexpz = 1 / expz;
    return (expz - RECexpz) / (expz + RECexpz);
}
inline Matrix_Double Tanh(const Matrix_Double& z) {
    return MapMatrix(z, TanhScalar);
}
inline static double TanhGradientScalar(double a) {
    return 1 - a * a;
}

inline Matrix_Double TanhGradient(const Matrix_Double& a) {
    return MapMatrix(a, TanhGradientScalar);
}

enum class ActivationType { None, Sigmoid, ReLU, Tanh };
inline static Matrix_Double Activate(const Matrix_Double& mat, ActivationType type) {
    switch (type)
    {
    case ActivationType::None:
        return mat;
    case ActivationType::Sigmoid:
        return Sigmoid(mat);
    case ActivationType::ReLU:
        return ReLU(mat);
    case ActivationType::Tanh:
        return Tanh(mat);
    default:
        break;
    }
    throw std::runtime_error("Unknow action type");
}
inline static std::vector<Matrix_Double> Activate(const std::vector<Matrix_Double>& mat, ActivationType type) {
    if (type == ActivationType::None) {
        return mat;
    }
    std::vector<Matrix_Double>copy(mat.size());
    switch (type)
    {
    case ActivationType::None:
        return mat;
    case ActivationType::Sigmoid:
        for (int i = 0;i < copy.size();i++) {
            copy[i] = Sigmoid(mat[i]);
        }
        return copy;
    case ActivationType::ReLU:
        for (int i = 0;i < copy.size();i++) {
            copy[i] = ReLU(mat[i]);
        }
        return copy;
    case ActivationType::Tanh:
        for (int i = 0;i < copy.size();i++) {
            copy[i] = Tanh(mat[i]);
        }
        return copy;
    default:
        break;
    }
    throw std::runtime_error("Unknow action type");
}
inline static Batch_Matrix Activate(const Batch_Matrix& bat, ActivationType type) {
    if (type == ActivationType::None) {
        return bat;
    }
    const std::vector<Matrix_Double>& mat = bat.GetInternalData();
    return Batch_Matrix(Activate(mat, type), bat.GetRowNum(), bat.GetColNum());
}
struct File_Handle_Inf
{
    bool is_good;
    int endIn;//本次操作于哪个字节结束
};
class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;   // 基类虚析构，否则 delete 派生类会漏析构
    /// <summary>前向：输入一批样本，返回本模块的输出。</summary>
    virtual std::vector<Matrix_Double> Forward(const std::vector<Matrix_Double>& x) = 0;
    /// <summary>反向：接收输出梯度，更新参数，返回输入的梯度。</summary>
    virtual std::vector<Matrix_Double> Back(double lr, const std::vector<Matrix_Double>& grad, double lambda = 0, double (*regular)(double) = nullptr) = 0;
    /// <summary>保存该节点
    virtual File_Handle_Inf Save(std::fstream& file, int beginIn) = 0;
    /// <summary>从文件中加载该节点
    virtual File_Handle_Inf Load(std::fstream& file, int beginIn) = 0;
};

enum class NeuralNodeType {
    MLP, //矩阵神经网络
    CNN,//单层的卷积神经网络
    MULCNN //多层卷积神经网络
};
/// <summary>神经网络的一层：持有本层的参数与前向缓存。</summary>
struct Layer_MLP {
    Matrix_Double W;                          // 权重矩阵，形状 out×in
    Matrix_Double b;                          // 偏置列向量，形状 out×1
    ActivationType act = ActivationType::None;// 本层激活函数

    // 前向传播时写入、反向传播时读取；保存权重时不必序列化这两项
    Matrix_Double z;                          // 激活前的线性输出 z = W·a_prev + b
    Matrix_Double a;                          // 激活后的输出 a = f(z)
};

class NeuralNetwork_MLP:public NeuralNetwork
{
public:
    NeuralNodeType type = NeuralNodeType::MLP;
    NeuralNetwork_MLP() = default;
    //调用此构造函数时act的大小应该大于等于lauers_Num的大小,若否会将最后一位设为None,为其余位置自动补上ReLU
    NeuralNetwork_MLP(std::vector<int>layers_ves, std::vector<ActivationType>acts_);
    File_Handle_Inf Load(std::fstream& file, int beginIn = 0) override;//通过读取文件中的内容，来得到一个神经网络    
    File_Handle_Inf Save(std::fstream& file, int beginIn = 0) override;//将神经网络存入一个文件中
    Matrix_Double Forward (const Matrix_Double& DataIn);//输入特征数据集，得到模型的预测数据集合
    Matrix_Double Back(double step_Length,const Matrix_Double& LossGradient,double lambda=0,double (*regular)(double)=nullptr);//输入调整的幅度与最后计算出的损失函数的梯度，逐步调整原函数，LossGradient表示∂L/∂a_out
    std::vector<Matrix_Double> Forward(const std::vector<Matrix_Double>& batch) override;
    std::vector<Matrix_Double> Back(double lr, const std::vector<Matrix_Double>& grad, double lambda = 0, double (*regular)(double) = nullptr) override;
private:
    std::vector<Layer_MLP> Layers_;
    int batch_col_buffer=10;
    int batch_row_buffer=10;
};

struct Layer_CNN
{
    Matrix_Double Ker;//卷积核
    ActivationType act = ActivationType::None;//激活层


    int p = 0;//padding,默认为0
    int s = 1;//stride，默认为1


    std::vector<Matrix_Double> z;//激活前的线性输出
    std::vector<Matrix_Double> a;//激活后的线性输出
};

class NeuralNetwork_CNN :public NeuralNetwork
{
public:
    NeuralNodeType type = NeuralNodeType::CNN;
    NeuralNetwork_CNN() = default;
    //输入核的size，acts不足会自动将最后一位设为None，其余填充ReLU，得到卷积神经网络
    NeuralNetwork_CNN(std::vector<int>Ker_Sizes, std::vector<ActivationType>acts_);
    void SetPadding(std::vector<int>paddings);
    void SetStride(std::vector<int>strides);
    File_Handle_Inf Load(std::fstream& file,int beginIn = 0) override;//通过读取文件中的内容，来得到一个神经网络    
    File_Handle_Inf Save(std::fstream& file,int beginIn = 0) override;//将神经网络存入一个文件中
    std::vector<Matrix_Double> Forward(const std::vector<Matrix_Double>& DataIn) override;//输入特征数据集，得到模型的预测数据集合
    std::vector<Matrix_Double> Back(double step_Length, const std::vector<Matrix_Double>& LossGradient, double lambda = 0, double (*regular)(double) = nullptr) override;//输入调整的幅度与最后计算出的损失函数的梯度，逐步调整原函数，LossGradient表示∂L/∂a_out
private:
    std::vector<Layer_CNN> Layers_ = {};
    std::vector<Matrix_Double> InputBuffer = {};//输入缓冲区用于存储输入
};


struct Layer_MULCNN
{
    Batch_Matrix Kers;//卷积核
    ActivationType act = ActivationType::None;//激活层


    int p = 0;//padding,默认为0
    int s = 1;//stride，默认为1
    int poll = 1;//poll,默认为1，表示池化相关的参数

    Batch_Matrix z;//激活前的线性输出
    Batch_Matrix a;//激活后的线性输出
    std::vector<AntiPoll_Data>antipoll_buffer;//反池化缓冲区
};

class NeuralNetwork_MULCNN :public NeuralNetwork
{
public:
    NeuralNodeType type = NeuralNodeType::MULCNN;

    NeuralNetwork_MULCNN() = default;
    //输入核的size，acts不足会自动将最后一位设为None，其余填充ReLU，得到卷积神经网络
    NeuralNetwork_MULCNN(std::vector<int>Layers_ves,std::vector<int> Ker_Sizes, std::vector<ActivationType> acts);
    void SetPadding(std::vector<int>paddings);
    void SetStride(std::vector<int>strides);
    void SetPoll(std::vector<int>polls);
    File_Handle_Inf Load(std::fstream& file, int beginIn = 0) override;//通过读取文件中的内容，来得到一个神经网络    
    File_Handle_Inf Save(std::fstream& file, int beginIn = 0) override;//将神经网络存入一个文件中
    std::vector<Matrix_Double> Forward(const std::vector<Matrix_Double>& DataIn) override;//输入特征数据集，得到模型的预测数据集合,其自动被转化为n*1 Batch_Matrix
    std::vector<Matrix_Double> Back(double step_Length, const std::vector<Matrix_Double>& LossGradient, double lambda = 0, double (*regular)(double) = nullptr) override;//输入调整的幅度与最后计算出的损失函数的梯度，逐步调整原函数，LossGradient表示∂L/∂a_out
private:
    std::vector<Layer_MULCNN> Layers_;
    Batch_Matrix InputBuffer;//输入缓冲区用于存储输入
    int outputColNum_Buffer = 1;
};



class Complex_NeuralNetwork {
public:
    Complex_NeuralNetwork() = default;
    Complex_NeuralNetwork(std::vector<std::shared_ptr<NeuralNetwork>> Nodes_);
    std::vector<Matrix_Double> Forward(const std::vector<Matrix_Double>& DataIn);
    std::vector<Matrix_Double> Back(double step_Length, const std::vector<Matrix_Double>& LossGradient, double lambda = 0, double (*regular)(double) = nullptr);
    File_Handle_Inf Load(std::fstream& file);
    File_Handle_Inf Save(std::fstream& file);

    // ===== 节点管理：添加 / 插入 / 删除 / 替换 / 清空 / 查询 =====

    /// <summary>在末尾添加一个节点。</summary>
    /// <param name="node">要添加的节点，不可为 nullptr</param>
    /// <exception cref="std::runtime_error">node 为 nullptr 时抛出</exception>
    void AddNode(const std::shared_ptr<NeuralNetwork>& node);

    /// <summary>在指定下标处插入一个节点，其后的节点依次后移。</summary>
    /// <param name="index">插入位置，取值范围 [0, GetNodeNum()]（等于个数表示插到末尾）</param>
    /// <param name="node">要插入的节点，不可为 nullptr</param>
    /// <exception cref="std::runtime_error">node 为 nullptr 或 index 越界时抛出</exception>
    void InsertNode(int index, const std::shared_ptr<NeuralNetwork>& node);

    /// <summary>删除指定下标的节点，其后的节点依次前移。</summary>
    /// <param name="index">删除位置，取值范围 [0, GetNodeNum()-1]</param>
    /// <exception cref="std::runtime_error">index 越界时抛出（含容器为空的情况）</exception>
    void DeleteNode(int index);

    /// <summary>替换指定下标的节点。</summary>
    /// <param name="index">替换位置，取值范围 [0, GetNodeNum()-1]</param>
    /// <param name="node">新节点，不可为 nullptr</param>
    /// <exception cref="std::runtime_error">node 为 nullptr 或 index 越界时抛出</exception>
    void SetNode(int index, const std::shared_ptr<NeuralNetwork>& node);

    /// <summary>清空所有节点。</summary>
    void ClearNodes();

    /// <summary>返回当前节点个数。</summary>
    /// <returns>节点个数</returns>
    int GetNodeNum() const;

    /// <summary>返回指定下标的节点（shared_ptr 副本，与容器内节点指向同一对象，非深拷贝）。</summary>
    /// <param name="index">下标，取值范围 [0, GetNodeNum()-1]</param>
    /// <returns>该节点的 shared_ptr</returns>
    /// <exception cref="std::runtime_error">index 越界时抛出</exception>
    std::shared_ptr<NeuralNetwork> GetNode(int index) const;
private:
    std::vector<std::shared_ptr<NeuralNetwork>> NeuralNodes;
};


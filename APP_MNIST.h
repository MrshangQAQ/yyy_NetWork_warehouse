#pragma once
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include "NeuralNetwork.h"
#include "Loss.h"   // 交叉熵/交叉熵梯度（与 MNIST 数据读写分离，CPU 演示工程也能用）


/// <summary>读取 MNIST 图像文件（IDX3 格式），一次性读入全部图像。</summary>
/// <param name="path">train-images-idx3-ubyte 文件的路径</param>
/// <returns>784×N 的矩阵，每列是一张图（符合项目"每列一个样本"的约定），像素已归一化到 [0,1]</returns>
/// <exception cref="std::runtime_error">文件无法打开，或魔数不是 2051 时抛出</exception>
Matrix_Double LoadMNISTImages(const std::string& path);

/// <summary>读取 MNIST 标签文件（IDX1 格式），一次性读入全部标签。</summary>
/// <param name="path">train-labels-idx1-ubyte 文件的路径</param>
/// <returns>长度为 N 的标签数组，每个元素取值 0~9</returns>
/// <exception cref="std::runtime_error">文件无法打开，或魔数不是 2049 时抛出</exception>
std::vector<unsigned char> LoadMNISTLabels(const std::string& path);

/// <summary>把 0~9 的标签数组转成 one-hot 矩阵。</summary>
/// <param name="labels">标签数组，每个元素取值 0~9</param>
/// <returns>10×N 的矩阵（每列一个样本）：标签为 k 的样本在第 k 行、第 i 列置 1.0，其余为 0</returns>
Matrix_Double LabelsToOneHot(const std::vector<unsigned char>& labels);

/// <summary>手写数字画板：提供 28×28 的绘制网格，并调用神经网络对手绘数字进行预测。</summary>
/// <remarks>画布约定：每个格子在窗口中为 20×20 像素，共 28×28 = 784 格；格子按行主序线性编号，索引 = 行号 × 28 + 列号。</remarks>
class APP_MNIST {
public:
	/// <summary>构造画板并创建窗口。</summary>
	/// <param name="ner">用于预测的神经网络引用；本类只持有引用、不管理其生命周期。</param>
	APP_MNIST(Complex_NeuralNetwork& ner);

	/// <summary>窗口是否仍然打开，供主循环判断是否继续运行。</summary>
	/// <returns>窗口打开返回 true，否则返回 false。</returns>
	bool IsRunning() const;

	/// <summary>处理本帧内到达的所有窗口事件（鼠标涂抹、空格预测、关闭窗口等）。</summary>
	void ProcessEvent();

	/// <summary>把当前画布内容渲染到窗口并显示。</summary>
	void Render();

private:
	static constexpr int kGrid = 28;            // 画布边长（格数）：28×28 与 MNIST 输入一致
	static constexpr int kCell = 20;            // 每格在窗口中的边长（像素）：20×28 = 560
	static constexpr int kSize = kGrid * kGrid; // 格子总数 784

	sf::RenderWindow window_;                   // 画板窗口
	std::vector<unsigned char> data_;           // 画布数据，长度 784：0=背景、255=笔画
	Matrix_Double Buffer_;					// 缓存数据，记录上一次预测得到的z
	sf::RectangleShape block_white_;            // 值为 0 时绘制的白色方块
	sf::RectangleShape block_black_;            // 值为 255 时绘制的黑色方块
	Complex_NeuralNetwork& network_;                // 用于预测的神经网络（外部注入，不拥有）

	/// <summary>把窗口内的鼠标像素坐标换算成画布格子的线性索引。</summary>
	/// <param name="position">鼠标在窗口中的像素坐标。</param>
	/// <returns>格子索引（0~783）；坐标落在画布之外时返回 -1。</returns>
	int Translate(sf::Vector2i position) const;

	/// <summary>在鼠标位置对应的格子上涂抹指定值（越界时忽略）。</summary>
	/// <param name="position">鼠标在窗口中的像素坐标。</param>
	/// <param name="value">涂抹的值：255 = 笔画、0 = 背景。</param>
	void PaintAt(sf::Vector2i position, unsigned char value);

	/// <summary>把当前画布送入神经网络，并在控制台打印各数字的概率。</summary>
	void Predict();

	/// <summary>清空画布（全部格子置 0）。</summary>
	void Clear();
	///手动反向传播,输入你期望的结果，将神经网络根据你期望的那个结果，。</summary>
	/// <summary>提示用户在控制台输入文件名，并把当前神经网络保存到该文件（文件不存在时自动创建）。</summary>
	/// <remarks>文件名为空视为取消；文件创建失败或写入失败时在控制台输出提示。</remarks>
	void SaveNetwork();
	/// 打印来自训练集的图形矩阵，检查实际上训练的输出是怎样的<summary>
public:
	void LoadPicture(Matrix_Double Picture_data);
private:
	// 事件兜底重载：visit 会把所有事件类型都派发过来，未特化的类型在此被忽略。
	void HandleEvent(const auto&) {}
	/// <summary>处理窗口关闭事件。</summary>
	void HandleEvent(const sf::Event::Closed&);
	/// <summary>处理鼠标按下事件：左键涂笔画、右键擦除。</summary>
	void HandleEvent(const sf::Event::MouseButtonPressed&);
	/// <summary>处理鼠标移动事件：按住按键拖动时连续涂抹。</summary>
	void HandleEvent(const sf::Event::MouseMoved&);
	/// <summary>处理按键事件：按下空格时进行预测，按 C 键清空画布，按 X 键手动反向传播，按 S 键保存神经网络。</summary>
	void HandleEvent(const sf::Event::KeyPressed&);
};

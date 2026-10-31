#include"APP_MNIST.h"
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <limits>

/// <summary>从二进制流中读取一个 4 字节的大端序无符号整数。</summary>
/// <param name="file">已经以二进制方式打开的输入流</param>
/// <returns>按大端序解析出的整数</returns>
/// <remarks>MNIST 的头部整数是大端序存储，而 x86 本机是小端序，所以必须逐字节手动拼装，不能直接 read 进 int。</remarks>
static uint32_t ReadBigEndianUInt32(std::fstream& file) {
	unsigned char b[4];
	file.read(reinterpret_cast<char*>(b), 4);
	// 大端序：b[0] 是最高字节，用「不断乘 256 再累加」的方式拼出整数
	uint32_t value = 0;
	value = value * 256 + b[0];
	value = value * 256 + b[1];
	value = value * 256 + b[2];
	value = value * 256 + b[3];
	return value;
}

Matrix_Double LoadMNISTImages(const std::string& path) {
	std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
	if (!file) {
		throw std::runtime_error("无法打开 MNIST 图像文件: " + path);
	}
	// 文件头共 16 字节：[魔数][图片数][行数][列数]
	uint32_t magic = ReadBigEndianUInt32(file);
	uint32_t count = ReadBigEndianUInt32(file);
	uint32_t rows = ReadBigEndianUInt32(file);
	uint32_t cols = ReadBigEndianUInt32(file);
	if (magic != 2051) {
		throw std::runtime_error("不是有效的 MNIST 图像文件（魔数错误）: " + path);
	}
	int pixels = (int)(rows * cols);
	// 一次性读入全部像素字节，避免反复访问磁盘
	std::vector<unsigned char> buf((size_t)count * (size_t)pixels);
	file.read(reinterpret_cast<char*>(buf.data()), buf.size());
	// 转成 pixels x count 的矩阵：第 p 行、第 i 列 = 第 i 张图的第 p 个像素
	std::vector<double> data((size_t)pixels * (size_t)count);
	for (uint32_t i = 0; i < count; i++) {
		for (int p = 0; p < pixels; p++) {
			// 矩阵按行主序存储，下标 = p * count + i；源数据是 buf[i * pixels + p]，归一化到 [0,1]
			data[(size_t)p * (size_t)count + (size_t)i] = buf[(size_t)i * (size_t)pixels + (size_t)p] / 255.0;
		}
	}
	return Matrix_Double(data, pixels, (int)count);
}

std::vector<unsigned char> LoadMNISTLabels(const std::string& path) {
	std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
	if (!file) {
		throw std::runtime_error("无法打开 MNIST 标签文件: " + path);
	}
	// 文件头共 8 字节：[魔数][标签数]
	uint32_t magic = ReadBigEndianUInt32(file);
	uint32_t count = ReadBigEndianUInt32(file);
	if (magic != 2049) {
		throw std::runtime_error("不是有效的 MNIST 标签文件（魔数错误）: " + path);
	}
	// 每个标签占 1 字节，取值 0~9
	std::vector<unsigned char> labels(count);
	file.read(reinterpret_cast<char*>(labels.data()), labels.size());
	return labels;
}

Matrix_Double LabelsToOneHot(const std::vector<unsigned char>& labels) {
	int n = (int)labels.size();
	// 先全部置 0，再按标签把对应位置置 1
	std::vector<double> data((size_t)10 * (size_t)n, 0.0);
	for (int i = 0; i < n; i++) {
		// 矩阵按行主序存储，第 labels[i] 行、第 i 列的下标 = labels[i] * n + i
		data[(size_t)labels[i] * (size_t)n + (size_t)i] = 1.0;
	}
	return Matrix_Double(data, 10, n);
}

/// <summary>构造画板并创建窗口。</summary>
APP_MNIST::APP_MNIST(Complex_NeuralNetwork& ner)
	: data_(kSize, 0), network_(ner),Buffer_(10,1) {
	window_.create(sf::VideoMode(sf::Vector2u((unsigned int)(kGrid * kCell), (unsigned int)(kGrid * kCell))), "MNIST Board");
	window_.setFramerateLimit(60);

	// 两种方块仅颜色不同，尺寸都是 kCell×kCell
	block_white_.setSize(sf::Vector2f((float)kCell, (float)kCell));
	block_white_.setFillColor(sf::Color::White);
	block_black_.setSize(sf::Vector2f((float)kCell, (float)kCell));
	block_black_.setFillColor(sf::Color::Black);
}

bool APP_MNIST::IsRunning() const {
	return window_.isOpen();
}

void APP_MNIST::ProcessEvent() {
	while (const auto event = window_.pollEvent()) {
		// SFML 3 用 visit 把事件派发给具体子类型；未特化的类型由兜底重载忽略
		event->visit([this](const auto& type) { this->HandleEvent(type); });
	}
}

void APP_MNIST::Render() {
	window_.clear(sf::Color::Black);
	for (int i = 0; i < kSize; i++) {
		// 行主序：第 i 格对应第 i/kGrid 行、第 i%kGrid 列，窗口坐标 = (列×kCell, 行×kCell)
		sf::Vector2f position((float)((i % kGrid) * kCell), (float)((i / kGrid) * kCell));
		// 约定：0 画白块、255 画黑块；意外中间值按 128 就近取舍
		sf::RectangleShape& block = (data_[i] > 128) ? block_white_ : block_black_;
		block.setFillColor(sf::Color(data_[i], data_[i], data_[i]));
		block.setPosition(position);
		window_.draw(block);
	}
	window_.display();
}

int APP_MNIST::Translate(sf::Vector2i position) const {
	int col = position.x / kCell;
	int row = position.y / kCell;
	// 鼠标可能移出画布（拖动时坐标为负或超出窗口），越界时返回 -1 由调用方忽略
	if (col < 0 || col >= kGrid || row < 0 || row >= kGrid) {
		return -1;
	}
	return row * kGrid + col;
}

void APP_MNIST::PaintAt(sf::Vector2i position, unsigned char value) {
	int index = Translate(position);
	if (index >= 0) {
		data_[index] = value;
	}
}

void APP_MNIST::Clear() {
	data_.assign(kSize, 0);
}

void APP_MNIST::SaveNetwork() {
	std::cout <<"please input the name of the file where you want save to(or only input 'enter' to cancel) "<< std::endl;
	std::string path;
	if (!std::getline(std::cin, path) || path.empty()) {
		std::cout <<"have canceled saving"<< std::endl;
		return;
	}
	std::fstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
	if (!file.is_open()) {
		std::cout << "couldn't gerenate the file" << path << std::endl;
		return;
	}
	bool ok = network_.Save(file).is_good;
	std::cout << (ok ? "save successfully" : "save failed") << path << std::endl;
}

void APP_MNIST::LoadPicture(Matrix_Double Picture_data) {
	for (int i = 0;i < data_.size();i++) {
		data_[i] = (int)(254.0 * Picture_data.GetInternalData()[i]);
	}
	Render();
}
void APP_MNIST::Predict() {
	// 画布 → 784×1 的输入矩阵（项目约定每列一个样本），并归一化到 [0,1] 与训练数据一致
	std::vector<double> input(kSize);
	for (int i = 0; i < kSize; i++) {
		input[i] = data_[i] / 255.0;
	}
	Matrix_Double x(input, 28, 28);
	// 输出层激活为 None：Forward 给出的是 logits，再经 Softmax 才是各数字的概率
	
	Buffer_ = GetFlattenMatrixFromBatch(network_.Forward({x}));

	std::cout << "text" << std::endl;

	Matrix_Double probs = Softmax(Buffer_);

	std::cout << "predict result(the probability of every numbers)" << std::endl;
	for (int i = 0; i < probs.GetRowNum(); i++) {
		std::cout << i << ": " << probs.GetData(i, 0) << std::endl;
	}
}

void APP_MNIST::HandleEvent(const sf::Event::Closed&) {
	window_.close();
}

void APP_MNIST::HandleEvent(const sf::Event::MouseButtonPressed& press) {
	if (press.button == sf::Mouse::Button::Left) {
		PaintAt(press.position, 255);   // 左键涂笔画
	}
	else if (press.button == sf::Mouse::Button::Right) {
		PaintAt(press.position, 0);     // 右键擦成背景
	}
}

void APP_MNIST::HandleEvent(const sf::Event::MouseMoved& move) {
	// 拖动涂抹：鼠标移动事件不带按键状态，用 isButtonPressed 轮询当前按住的键
	if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
		PaintAt(move.position, 255);
	}
	else if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) {
		PaintAt(move.position, 0);
	}
}

void APP_MNIST::HandleEvent(const sf::Event::KeyPressed& press) {
	if (press.code == sf::Keyboard::Key::Space) {
		Predict();
	}
	else if (press.code == sf::Keyboard::Key::C) {
		Clear();
	}
	else if (press.code == sf::Keyboard::Key::S) {
		SaveNetwork();
	}
}

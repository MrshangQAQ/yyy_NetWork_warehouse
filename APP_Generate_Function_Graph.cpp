#include"APP_Generate_Function_Graph.h"

APP_Generate_Function_Graph::APP_Generate_Function_Graph(float x, float y, unsigned int FramerateLimit){
	window_.create(sf::VideoMode(sf::Vector2u((unsigned int)x,(unsigned int)y)), "SFML work");
	window_.setFramerateLimit(FramerateLimit);

	BackGround_.setSize(sf::Vector2f(x, y));
	BackGround_.setFillColor(sf::Color::White);
	BackGround_.setPosition(sf::Vector2f(0,0));

	Grid_x.setSize(sf::Vector2f(2*x, 2.f));
	Grid_y.setSize(sf::Vector2f(2.f, 2*y));
	Grid_x.setFillColor(sf::Color::Blue);
	Grid_y.setFillColor(sf::Color::Blue);

	Grid_x_main.setSize(sf::Vector2f(2*x, 4));
	Grid_y_main.setSize(sf::Vector2f(4, 2 * y));
	Grid_x_main.setFillColor(sf::Color::Black);
	Grid_y_main.setFillColor(sf::Color::Black);

	Grid_x_main.setPosition(sf::Vector2f(0, y / 2 ));
	Grid_y_main.setPosition(sf::Vector2f(x / 2, 0));
}
bool APP_Generate_Function_Graph::IsRunning() const {
	return window_.isOpen();
}
void APP_Generate_Function_Graph::ProcessEvent() {
	while (const auto event = window_.pollEvent()) {
		event->visit([this](const auto& type) {this->HandleEvent(type);});
	}
}
void APP_Generate_Function_Graph::Render() {
	window_.draw(BackGround_);
	window_.draw(Grid_x_main);
	window_.draw(Grid_y_main);
	sf::Vector2f window_center(window_.getSize().x / 2, window_.getSize().y / 2);
	for (int i = 0;i <window_.getSize().y / Grid_length;i++) {
		Grid_x.setPosition(sf::Vector2f(0, window_center.y + i * Grid_length));
		window_.draw(Grid_x);
		Grid_x.setPosition(sf::Vector2f(0, window_center.y - i * Grid_length));
		window_.draw(Grid_x);
	}
	for (int i = 0;i < window_.getSize().x / Grid_length;i++) {
		Grid_y.setPosition(sf::Vector2f(window_center.x + i * Grid_length, 0));
		window_.draw(Grid_y);
		Grid_y.setPosition(sf::Vector2f(window_center.x - i * Grid_length, 0));
		window_.draw(Grid_y);
	}
	for (int i = 0;i < functionGraph_.size();i++) {
		window_.draw(functionGraph_[i]);
	}
	window_.display();
}

int APP_Generate_Function_Graph::AddFunction(Function_Node function, double accuracy, sf::Color color) {
	int pointNums_x =window_.getSize().x/(ZoomFactor_x * accuracy);
	functionGraph_.push_back(sf::VertexArray(sf::PrimitiveType::LineStrip));
	functions_.push_back(function);
	accuracys_.push_back(accuracy);
	for (int i = -pointNums_x;i < pointNums_x;i++) {
		(functionGraph_.back()).append(sf::Vertex{ Translate(sf::Vector2f(i * accuracy,function(i * accuracy))), color });
	}
	return functionGraph_.size() - 1;
}
int APP_Generate_Function_Graph::AddDifferentialEquation(MulFunction_Node DifExp, double accuracy, sf::Vector2f beginPoint, int SimilarTimes, sf::Color color) {
	std::vector<double> point = { beginPoint.x,beginPoint.y };
	std::vector<double> Direction = { 0,0 };
	std::vector<double> OldDirection = { 0,0 };
	functionGraph_.push_back(sf::VertexArray(sf::PrimitiveType::LineStrip));
	for (int i = 0;i < SimilarTimes;i++) {
		(functionGraph_.back()).append(sf::Vertex{ Translate(sf::Vector2f(point[0],point[1])), color});
		Direction[1] = DifExp(point);
		Direction[0] = 1 / std::sqrt(Direction[1] * Direction[1] + 1);
		Direction[1] *= Direction[0];
		if (Direction[0] * OldDirection[0] + Direction[1] * OldDirection[1] < 0) {
			Direction[0] *= -1;
			Direction[1] *= -1;
		}
		OldDirection = Direction;
		point[0] += accuracy * Direction[0];
		point[1] += accuracy * Direction[1];
	}
	point = { beginPoint.x,beginPoint.y };
	Direction = { 0,0 };
	OldDirection = { 0,0 };
	functionGraph_.push_back(sf::VertexArray(sf::PrimitiveType::LineStrip));
	for (int i = 0;i < SimilarTimes;i++) {
		(functionGraph_.back()).append(sf::Vertex{ Translate(sf::Vector2f(point[0],point[1])), color });
		Direction[1] = DifExp(point);
		Direction[0] = 1 / std::sqrt(Direction[1] * Direction[1] + 1);
		Direction[1] *= Direction[0];
		if (Direction[0] * OldDirection[0] + Direction[1] * OldDirection[1] < 0) {
			Direction[0] *= -1;
			Direction[1] *= -1;
		}
		OldDirection = Direction;
		point[0] -= accuracy * Direction[0];
		point[1] -= accuracy * Direction[1];
	}
	return functionGraph_.size();
}
void APP_Generate_Function_Graph::SetZoomFactor(double x, double y){
	ZoomFactor_x = x;
	ZoomFactor_y = y;
}
void APP_Generate_Function_Graph::HandleEvent(const auto& event) {

}
void APP_Generate_Function_Graph::HandleEvent(sf::Event::Closed close) {
	window_.close();
}
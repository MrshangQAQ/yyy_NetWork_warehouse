#pragma once
#include <SFML/Graphics.hpp>
#include"MathFunction.h"
class APP_Generate_Function_Graph
{
public:
	APP_Generate_Function_Graph(float x, float y, unsigned int FramerateLimit);
	int AddFunction(Function_Node function,double accuracy,sf::Color color = sf::Color::Black);
	int AddDifferentialEquation(MulFunction_Node DifExp,double accuracy, sf::Vector2f beginPoint = sf::Vector2f(0, 0),int SimilarTimes=1000, sf::Color color = sf::Color::Black);
	bool IsRunning() const;
	void ProcessEvent();
	void Render();
	void SetZoomFactor(double x,double y);
	std::vector<Function_Node> functions_;
	std::vector<sf::VertexArray> functionGraph_;
	std::vector<double> accuracys_;
private:
	sf::RenderWindow window_;
	sf::RectangleShape BackGround_;
	sf::RectangleShape Grid_x;
	sf::RectangleShape Grid_y;
	sf::RectangleShape Grid_x_main;
	sf::RectangleShape Grid_y_main;
	double Grid_length = 80;
	double ZoomFactor_x = 80;
	double ZoomFactor_y = 80;
	sf::Vector2f Translate(sf::Vector2f oldPositon) {
		return sf::Vector2f((ZoomFactor_x * oldPositon.x + window_.getSize().x / 2),(-ZoomFactor_y * oldPositon.y + window_.getSize().y/2));
	}
private:
	void HandleEvent(const auto& event);
	void HandleEvent(sf::Event::Closed close);
};
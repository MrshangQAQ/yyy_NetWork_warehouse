#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "Complex.h"
struct PolyDivison_Result;
struct PolyGDC_Result;
struct PolyView {
	const Complex* first;
	size_t len;

	PolyView(const std::vector<Complex>& v) : first(v.data()), len(v.size()) {}
	PolyView(const Complex* f, size_t n) : first(f), len(n) {}

	PolyView LowHalf()  const { return PolyView(first, len / 2); }
	PolyView HighHalf() const { return PolyView(first + len / 2, len - len / 2); }

	Complex operator[](size_t i) const { return first[i]; }
};
struct PolyView_Double {
	const Complex_Double* first;
	size_t len;

	PolyView_Double(const std::vector<Complex_Double>& v) : first(v.data()), len(v.size()) {}
	PolyView_Double(const Complex_Double* f, size_t n) : first(f), len(n) {}

	PolyView_Double LowHalf()  const { return PolyView_Double(first, len / 2); }
	PolyView_Double HighHalf() const { return PolyView_Double(first + len / 2, len - len / 2); }

	Complex_Double operator[](size_t i) const { return first[i]; }
};

class Polynomial
{
public:
	std::vector<Complex> Coefficient;
	Complex CalculatedValue(Complex val) const;
	Complex_Double CalculatedValue (Complex_Double val) const;
	Polynomial(const PolyView View) {
		for (int i = 0;i < View.len;i++) {
			Coefficient.push_back(*(View.first + i));
		}
		Sort();
	}
	Polynomial(std::vector<Complex> Coef = {}) { 
		Coefficient = Coef; 
		Sort();
	};
	Polynomial(Complex constant) {
		Coefficient = { constant };
		Sort();
	}
	PolyView View() const { return PolyView(Coefficient); }
	void Differentiation();
	void Sort();//去掉末尾为0的部分;
	bool IsEmpty() const;
};
Polynomial operator + (const Polynomial & pol1, const Polynomial & pol2);
Polynomial& operator += (Polynomial& pol1, const Polynomial& pol2);
Polynomial operator - (const Polynomial& pol1, const Polynomial& pol2);
Polynomial& operator -= (Polynomial& pol1, const Polynomial& pol2);
Polynomial operator * (Complex com, const Polynomial& pol);
Polynomial& operator *= (Polynomial& pol, const Complex& com);
Polynomial operator>>(const Polynomial& pol, unsigned int index);//让多项式乘以index个x
Polynomial& operator>>=(Polynomial& pol, unsigned int index);//让多项式乘以index个x
Polynomial operator<<(const Polynomial& pol, unsigned int index);//让多项式除以index个x，次数小于0的部分会被舍去
Polynomial& operator<<=(Polynomial& pol, unsigned int index);//让多项式除以index个x，次数小于0的部分会被舍去
Polynomial operator * (const Polynomial& pol1, const Polynomial& pol2);
Polynomial& operator *= (Polynomial& pol1, const Polynomial& pol2);
PolyDivison_Result operator/(Polynomial pol1,const Polynomial& pol2);//这里直接按照值传算了
Polynomial operator/(const Polynomial& pol1, const Complex& factor);
Polynomial& operator/=(Polynomial& pol1, const Complex& factor);
std::ostream& operator<<(std::ostream& os, const Polynomial& pol);

struct PolyDivison_Result {
	Polynomial Quotient;
	Polynomial Remainder;
};
struct PolyGDC_Result {
	Polynomial GDC = Polynomial(Complex(0, 0));
	Polynomial R1 = Polynomial(Complex(0, 0));
	Polynomial R2 = Polynomial(Complex(0, 0));
};



PolyGDC_Result GetGCD(Polynomial pol1,Polynomial pol2);
bool IsExactlyDivisible(Polynomial pol1, Polynomial pol2);
void Swap(Polynomial& com1, Polynomial& com2);
void NaiveConvolve(const PolyView& a, const PolyView& b, std::vector<Complex>& result);
Polynomial KaratsubaMul(const PolyView& a,const PolyView& b, int threshold);
/// <summary>原地快速傅里叶变换（迭代版 Cooley-Tukey），长度必须是 2 的幂</summary>
/// <param name="a">待变换数组，结果原地写回</param>
/// <param name="inverse">false = 正向变换；true = 逆变换（最后自动除以 n 归一化）</param>
void FFT(std::vector<Complex_Double>& a, bool inverse);

/// <summary>用 FFT 计算两个多项式的乘积（内部 double 工作区，出口舍入回有理数）</summary>
/// <param name="a">第一个多项式视图</param>
/// <param name="b">第二个多项式视图</param>
/// <returns>乘积多项式</returns>
Polynomial FFTMul(const Polynomial& a, const Polynomial& b);
Polynomial GetDifferentiation(Polynomial pol);
int Complexity_Comparison(const Polynomial& a, const Polynomial& b);//返回1表示左边更简单，返回-1表示右边更简单，返回0表示平局
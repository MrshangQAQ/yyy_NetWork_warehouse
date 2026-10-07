#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include "Rational.h"
class Complex_Double;
class Complex {
public:
	Rational real;
	Rational imag;
	Complex(Rational r = Rational(), Rational i = Rational()) { real = r;imag = i; };
	Complex(double r, double i = 0) { real = Rational(r); imag = Rational(i); };
	Complex(const Complex& other) { real = other.real; imag = other.imag; };
	Complex(Complex_Double com_d);//从浮点复数转换；函数体要访问 Complex_Double 的成员，定义放在其完整定义之后
	const Complex Conjugate() const;
	double Modulus() const;
	Rational ModulusSquare() const;
};
Complex operator""_i(long double val);
Complex operator""_i(unsigned long long val);
const bool operator==(const Complex& other1, const Complex& other2);
const bool operator!=(const Complex& other1, const Complex& other2);
Complex operator+(const Complex& other1, const Complex& other2);
Complex& operator+=(Complex& other1, const Complex& other2);
Complex operator-(const Complex& other1, const Complex& other2);
Complex operator-(const Complex & com);
Complex& operator-=(Complex& other1, const Complex& other2);
Complex operator*(const Complex& other1, const Complex& other2);
Complex& operator*=(Complex& other1, const Complex& other2);
Complex operator/(const Complex& other1, const float& other2);
Complex operator/(const Complex& other1, const Complex& other2);
Complex& operator/=(Complex& other1, const Complex& other2);
Complex operator^(const Complex& com, const int& index);
std::ostream& operator<<(std::ostream& os, const Complex& complex);

void Swap(Complex& com1, Complex& com2);

class Complex_Double
{
public:
	double real;
	double imag;
	Complex_Double(double real_=0, double imag_=0) { real = real_;imag = imag_; }
	Complex_Double(Complex complex) { real = (double)complex.real, imag = (double)complex.imag; }
	const Complex_Double Conjugate() const { return Complex_Double(real,-imag); }
	double ModulusSquare() const { return real * real + imag * imag; }
};
/// <summary>Complex 从浮点复数的转换构造函数：定义必须放在 Complex_Double 完整定义之后，
/// 因为函数体要访问 com_d.real / com_d.imag，而 Complex 类定义处它还只是前置声明（不完整类型）</summary>
inline Complex::Complex(Complex_Double com_d) { real = com_d.real; imag = com_d.imag; }
inline Complex_Double& operator+=(Complex_Double& comd1, const Complex_Double& comd2) { comd1.real += comd2.real;comd1.imag += comd2.imag;return comd1; }
inline Complex_Double operator+(Complex_Double comd1, const Complex_Double& comd2) { comd1 += comd2;return comd1; }
inline Complex_Double& operator-=(Complex_Double& comd1, const Complex_Double& comd2) { comd1.real -= comd2.real;comd1.imag -= comd2.imag;return comd1; }
inline Complex_Double operator-(Complex_Double comd1, const Complex_Double& comd2) { comd1 -= comd2;return comd1; }
inline Complex_Double operator-(const Complex_Double& comd) { return Complex_Double(-comd.real, -comd.imag); }
inline Complex_Double& operator*=(Complex_Double& comd1, const Complex_Double& comd2) {
	double oldreal = comd1.real;
	comd1.real*= comd2.real;
	comd1.real -= comd2.imag * comd1.imag;
	comd1.imag*= comd2.real;
	comd1.imag += oldreal * comd2.imag;
	return comd1; 
}
inline Complex_Double operator*(Complex_Double comd1, const Complex_Double& comd2) { comd1 *= comd2;return comd1; }
inline Complex_Double& operator/=(Complex_Double& comd1, const Complex_Double& comd2) {
	comd1*=comd2.Conjugate();
	double tmp = comd2.ModulusSquare();
	comd1.real /= tmp;
	comd1.imag /= tmp;
	return comd1;
}
inline Complex_Double operator/(Complex_Double comd1, const Complex_Double& comd2) { comd1 /= comd2;return comd1; }
inline std::ostream& operator<<(std::ostream& os, const Complex_Double& complex) {
	if (complex.real == 0 && complex.imag != 0) {
		os << complex.imag << "i";
		return os;
	}
	os << complex.real;
	if (complex.imag > 0) {
		os << "+" << complex.imag << "i";
	}
	else if (complex.imag < 0) {
		os << complex.imag << "i";
	}
	return os;
}
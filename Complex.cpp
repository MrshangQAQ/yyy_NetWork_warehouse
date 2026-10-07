#include"Complex.h"
const Complex Complex::Conjugate() const {
	return Complex(real, -imag);
}
double Complex::Modulus() const {
	return sqrt((double)ModulusSquare());
}
Complex operator""_i(long double val) {
	return Complex(0, (double)val);
}
Complex operator""_i(unsigned long long val) {
	return Complex(0, (long long)val);
}
const bool operator==(const Complex& other1, const Complex& other2) {
	return other1.real == other2.real && other1.imag == other2.imag;
}
const bool operator!=(const Complex& other1, const Complex& other2) {
	return !(other1 == other2);
}
Complex operator+(const Complex& other1, const Complex& other2) {
	return Complex(other1.real + other2.real, other1.imag + other2.imag);
}
Complex& operator+=(Complex& other1, const Complex& other2) {
	other1.real += other2.real;
	other1.imag += other2.imag;
	return other1;
}
Complex operator-(const Complex& other1, const Complex& other2) {
	return Complex(other1.real - other2.real, other1.imag - other2.imag);
}
Complex operator-(const Complex& com) {
	return Complex(-com.real, -com.imag);
}
Complex& operator-=(Complex& other1, const Complex& other2) {
	other1.real -= other2.real;
	other1.imag -= other2.imag;
	return other1;
}
Complex operator*(const Complex& other1, const Complex& other2) {
	return Complex(other1.real * other2.real - other1.imag * other2.imag, other1.real * other2.imag + other1.imag * other2.real);
}
Complex& operator*=(Complex& other1, const Complex& other2) {
	Rational o1_oldreal = other1.real;
	other1.real = other1.real * other2.real - other1.imag * other2.imag;
	other1.imag = o1_oldreal * other2.imag + other1.imag * other2.real;
	return other1;
}
Complex operator/(const Complex& other1, const float& other2) {
	Rational d((double)other2);
	return Complex(other1.real / d, other1.imag / d);
}
Complex operator/(const Complex& other1, const Complex& other2) {
	Complex numerator = other1 * other2.Conjugate();
	Rational denom = other2.ModulusSquare();
	return Complex(numerator.real / denom, numerator.imag / denom);
}
Complex& operator/=(Complex& other1, const Complex& other2) {
	other1 = other1 / other2;
	return other1;
}
Complex operator^(const Complex& com, const int& index) {
	Complex tmp(1, 0);
	if (index == 0) {
		return tmp;
	}
	if (index < 0) {
		for (int i = 0;i > index;i--) {
			tmp = tmp / com;
		}
		return tmp;
	}
	if (index > 0) {
		for (int i = 0;i < index;i++) {
			tmp = tmp * com;
		}
		return tmp;
	}
}
std::ostream& operator<<(std::ostream& os, const Complex& complex) {
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

Rational Complex::ModulusSquare() const {
	return imag * imag + real * real;
}

void Swap(Complex& com1, Complex& com2) {
	if (&com1 == &com2) {
		return;
	}
	Complex tmp(com1);
	com1 = com2;
	com2 = tmp;
}
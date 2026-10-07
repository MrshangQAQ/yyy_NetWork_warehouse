#pragma once
#include <iostream>
#include <numeric>    // std::gcd
#include <stdexcept>
#include <cmath>

/// <summary>有理数类：以「分子/分母」两个 long long 整数精确表示分数，构造与每次运算后自动约分。
/// 分母恒为正，符号统一放在分子上；分母为零时抛出异常。</summary>
class Rational {
public:
	long long num;   ///< 分子（可为负）
	long long den;   ///< 分母（恒为正，已约分）

	/// <summary>从整数/分数构造，自动约分并规范化（d 为负时符号移到分子）</summary>
	/// <param name="n">分子</param>
	/// <param name="d">分母，不能为 0</param>
	/// <exception cref="std::runtime_error">分母为 0 时抛出</exception>
	Rational(long long n = 0, long long d = 1) : num(n), den(d) {
		if (den == 0) throw std::runtime_error("Rational: denominator is zero");
		Normalize();
	}

	/// <summary>从浮点数构造：四舍五入到 10^-8 精度后转为分数（如 0.1 → 1/10）。
	/// 注意：无法精确表示 1/3 这类无限小数，仅用于兼容浮点输入；精确计算请直接用分数构造。</summary>
	/// <param name="d">浮点数值（绝对值不宜超过 9.2e10，否则 long long 溢出）</param>
	Rational(double d) {
		long long scaled = (long long)std::llround(d * 100000000.0);
		num = scaled; den = 100000000LL;
		Normalize();
	}

	/// <summary>约分并规范化：分母恒为正，分子分母互素，零统一表示为 0/1</summary>
	void Normalize() {
		if (den == 0) throw std::runtime_error("Rational: denominator is zero");
		if (den < 0) { num = -num; den = -den; }
		long long g = std::gcd(num < 0 ? -num : num, den);
		if (g > 1) { num /= g; den /= g; }
		if (num == 0) den = 1;
	}

	/// <summary>转为 double（近似值，仅用于输出/浮点兼容）。
	/// 声明为 explicit：若允许隐式转换，内建 (double,double) 运算符会与自定义混合比较
	/// 形成重载歧义（C2666）。</summary>
	explicit operator double() const { return (double)num / (double)den; }
};

/// <summary>加法：a/b + c/d = (ad+bc)/bd，构造时自动约分</summary>
inline Rational operator+(const Rational& a, const Rational& b) {
	return Rational(a.num * b.den + b.num * a.den, a.den * b.den);
}
/// <summary>减法</summary>
inline Rational operator-(const Rational& a, const Rational& b) {
	return Rational(a.num * b.den - b.num * a.den, a.den * b.den);
}
/// <summary>乘法：分子分母分别相乘，构造时自动约分</summary>
inline Rational operator*(const Rational& a, const Rational& b) {
	return Rational(a.num * b.num, a.den * b.den);
}
/// <summary>除法：乘以倒数，除数为 0 时构造抛出异常</summary>
inline Rational operator/(const Rational& a, const Rational& b) {
	return Rational(a.num * b.den, a.den * b.num);
}
/// <summary>一元负号</summary>
inline Rational operator-(const Rational& a) { return Rational(-a.num, a.den); }
/// <summary>自加：a += b</summary>
inline Rational& operator+=(Rational& a, const Rational& b) { a = a + b; return a; }
/// <summary>自减：a -= b</summary>
inline Rational& operator-=(Rational& a, const Rational& b) { a = a - b; return a; }
/// <summary>自乘：a *= b</summary>
inline Rational& operator*=(Rational& a, const Rational& b) { a = a * b; return a; }
/// <summary>自除：a /= b</summary>
inline Rational& operator/=(Rational& a, const Rational& b) { a = a / b; return a; }

/// <summary>相等比较（有理数已约分，直接比较分子分母）</summary>
inline bool operator==(const Rational& a, const Rational& b) { return a.num == b.num && a.den == b.den; }
/// <summary>不等比较</summary>
inline bool operator!=(const Rational& a, const Rational& b) { return !(a == b); }
/// <summary>大小比较：交叉相乘（a/b &lt; c/d ⇔ ad &lt; bc，b、d 恒为正）</summary>
inline bool operator<(const Rational& a, const Rational& b) { return a.num * b.den < b.num * a.den; }
/// <summary>大于比较</summary>
inline bool operator>(const Rational& a, const Rational& b) { return b < a; }
/// <summary>小于等于比较</summary>
inline bool operator<=(const Rational& a, const Rational& b) { return !(b < a); }
/// <summary>大于等于比较</summary>
inline bool operator>=(const Rational& a, const Rational& b) { return !(a < b); }

// ---- 与 double 的混合比较（Matrix 中 Rational 与 float approximation 比较用） ----
/// <summary>混合比较：Rational 与 double（按 double 近似值比较）</summary>
inline bool operator<(const Rational& a, double b) { return (double)a < b; }
inline bool operator<(double a, const Rational& b) { return a < (double)b; }
inline bool operator>(const Rational& a, double b) { return (double)a > b; }
inline bool operator>(double a, const Rational& b) { return a > (double)b; }
inline bool operator<=(const Rational& a, double b) { return (double)a <= b; }
inline bool operator<=(double a, const Rational& b) { return a <= (double)b; }
inline bool operator>=(const Rational& a, double b) { return (double)a >= b; }
inline bool operator>=(double a, const Rational& b) { return a >= (double)b; }
inline bool operator==(const Rational& a, double b) { return (double)a == b; }
inline bool operator==(double a, const Rational& b) { return a == (double)b; }
inline bool operator!=(const Rational& a, double b) { return (double)a != b; }
inline bool operator!=(double a, const Rational& b) { return a != (double)b; }

/// <summary>输出：形如 num/den（den 为 1 时只输出 num）</summary>
inline std::ostream& operator<<(std::ostream& os, const Rational& r) {
	os << r.num;
	if (r.den != 1) os << "/" << r.den;
	return os;
}

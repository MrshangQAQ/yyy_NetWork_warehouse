#include"Polynomial.h"
#include <cmath>

Complex Polynomial::CalculatedValue(Complex val) const {
	Complex mul = 1;
	Complex sum = 0;
	for (int i = 0;i < Coefficient.size();i++) {
		sum += mul * Coefficient[i];
		mul *= val;
	}
	return sum;
}
Complex_Double Polynomial::CalculatedValue(Complex_Double val) const {
	Complex_Double mul = 1;
	Complex_Double sum = 0;
	for (int i = 0;i < Coefficient.size();i++) {
		sum += mul * Complex_Double(Coefficient[i]);
		mul *= val;
	}
	return sum;
}
void Polynomial::Sort() {
	while (!IsEmpty()&&Coefficient.back() == Complex(0,0)) {
		Coefficient.pop_back();
	}
}
bool Polynomial::IsEmpty() const {
	return Coefficient.size() == 0;
}
void Polynomial::Differentiation() {
	for (int i = 0;i < Coefficient.size();i++) {
		Coefficient[i] *= i;
	}
	*this <<= 1;
	Sort();
}
Polynomial GetDifferentiation(Polynomial pol) {
	pol.Differentiation();
	return pol;
}
Polynomial operator + (const Polynomial& pol1, const Polynomial& pol2){
	Polynomial newpol;
	for (int i = 0;i < pol1.Coefficient.size();i++) {
		newpol.Coefficient.push_back(pol1.Coefficient[i]);
	}
	for (int i = 0;i < pol2.Coefficient.size();i++) {
		if (i < newpol.Coefficient.size()) {
			newpol.Coefficient[i] += pol2.Coefficient[i];
		}
		else {
			newpol.Coefficient.push_back(pol2.Coefficient[i]);
		}
	}
	newpol.Sort();
	return newpol;
}
Polynomial& operator += (Polynomial& pol1, const Polynomial& pol2) {
	for (int i = 0;i < pol2.Coefficient.size();i++) {
		if (i < pol1.Coefficient.size()) {
			pol1.Coefficient[i] += pol2.Coefficient[i];
		}
		else {
			pol1.Coefficient.push_back(pol2.Coefficient[i]);
		}
	}
	pol1.Sort();
	return pol1;
}
Polynomial operator - (const Polynomial& pol1, const Polynomial& pol2) {
	Polynomial newpol;
	for (int i = 0;i < pol1.Coefficient.size();i++) {
		newpol.Coefficient.push_back(pol1.Coefficient[i]);
	}
	for (int i = 0;i < pol2.Coefficient.size();i++) {
		if (i < newpol.Coefficient.size()) {
			newpol.Coefficient[i] -= pol2.Coefficient[i];
		}
		else {
			newpol.Coefficient.push_back(-pol2.Coefficient[i]);
		}
	}
	newpol.Sort();
	return newpol;
}
Polynomial& operator -= (Polynomial& pol1, const Polynomial& pol2) {
	for (int i = 0;i < pol2.Coefficient.size();i++) {
		if (i < pol1.Coefficient.size()) {
			pol1.Coefficient[i] -= pol2.Coefficient[i];
		}
		else {
			pol1.Coefficient.push_back(-pol2.Coefficient[i]);
		}
	}
	pol1.Sort();
	return pol1;
}

Polynomial operator * (Complex com, const Polynomial& pol) {
	Polynomial newpol;
	for (int i = 0;i < pol.Coefficient.size();i++) {
		newpol.Coefficient.push_back(com * pol.Coefficient[i]);
	}
	newpol.Sort();
	return newpol;
}

Polynomial& operator *= (Polynomial& pol, const Complex& com){
	for (int i = 0;i < pol.Coefficient.size();i++) {
		pol.Coefficient[i] *= com;
	}
	pol.Sort();
	return pol;
}
Polynomial operator>>(const Polynomial& pol, unsigned int index) {
	if (index == 0) {
		return pol;
	}
	Polynomial newpol;
	for (int i = 0;i < index;i++) {
		newpol.Coefficient.push_back(Complex(0, 0));
	}
	for (int i = 0;i < pol.Coefficient.size();i++) {
		newpol.Coefficient.push_back(pol.Coefficient[i]);
	}
	newpol.Sort();
	return newpol;
}
Polynomial& operator>>=(Polynomial& pol, unsigned int index) {
	int oldsize = pol.Coefficient.size();
	for (unsigned int i = 0;i < index;i++) {
		pol.Coefficient.push_back(Complex(0, 0));
	}
	for (unsigned int i = oldsize - 1;i >= 0;i--) {
		if (pol.Coefficient[i] == Complex(0, 0)) {
			continue;
		}
		pol.Coefficient[i + index] = pol.Coefficient[i];
		pol.Coefficient[i] = Complex(0, 0);
	}
	pol.Sort();
	return pol;
}

Polynomial operator<<(const Polynomial& pol, unsigned int index) {
	Polynomial newpol;
	if (index == 0) {
		newpol = pol;
		return newpol;
	}
	for (int i = index;i < pol.Coefficient.size();i++) {
		newpol.Coefficient.push_back(pol.Coefficient[i]);
	}
	newpol.Sort();
	return newpol;
}

Polynomial& operator<<=(Polynomial& pol, unsigned int index) {
	int endsize = (index >= pol.Coefficient.size()) ? 0
		: (int)(pol.Coefficient.size() - index);
	for (int i = 0;i < endsize;i++) {
		pol.Coefficient[i] = pol.Coefficient[i + index];
	}
	pol.Coefficient.resize(endsize);
	return pol;
}

Polynomial operator * (const Polynomial& pol1, const Polynomial& pol2) {
	/*
	if (pol1.Coefficient.size() > 512 && pol2.Coefficient.size() > 512) {
		return FFTMul(pol1.View(),pol2.View());
	}
	*/
	Polynomial a = pol1, b = pol2;
	size_t n = std::max(a.Coefficient.size(), b.Coefficient.size());
	a.Coefficient.resize(n, Complex(0, 0));   // 短的补零，保证两边一样长
	b.Coefficient.resize(n, Complex(0, 0));
	return KaratsubaMul(a.View(), b.View(), 128);
}
Polynomial& operator *= (Polynomial& pol1, const Polynomial& pol2) {
	Polynomial oldpol1 = pol1;
	pol1.Coefficient.clear();
	for (int i = 0;i < pol2.Coefficient.size();i++) {
		if (pol2.Coefficient[i] != Complex(0, 0)) {
			pol1 += pol2.Coefficient[i] * (oldpol1 >> i);
		}
	}
	pol1.Sort();
	return pol1;
}
PolyDivison_Result operator/(Polynomial pol1,const Polynomial& pol2) {
	if (pol1.Coefficient.size() < pol2.Coefficient.size()) {
		return PolyDivison_Result{ Polynomial(Complex(0,0)),pol1 };
	}
	Polynomial QuoPol;
	Complex invLead = Complex(1, 0) / pol2.Coefficient.back();
	int diff = (int)pol1.Coefficient.size() - (int)pol2.Coefficient.size();
	QuoPol.Coefficient.resize(diff+1,Complex(0,0));
	for (int step = 0; step <= diff; step++) {
		Complex factor = pol1.Coefficient.back() * invLead;
		int start = (int)pol1.Coefficient.size() - (int)pol2.Coefficient.size();
		for (int j = 0; j < (int)pol2.Coefficient.size(); j++) {
			pol1.Coefficient[start + j] -= factor * pol2.Coefficient[j];
		}
		QuoPol.Coefficient[diff - step] = factor;
		pol1.Coefficient.pop_back();
	}
	pol1.Sort();
	return PolyDivison_Result{ QuoPol,pol1};
}
Polynomial operator/(const Polynomial& pol1, const Complex& factor) {
	std::vector<Complex> newData;
	Complex Fac = Complex(1, 0) / factor;
	for (int i = 0;i < pol1.Coefficient.size();i++) {
		newData.push_back(Fac * pol1.Coefficient[i]);
	}
	return Polynomial(newData);
}
Polynomial& operator/=(Polynomial& pol1, const Complex& factor) {
	Complex Fac = Complex(1, 0) / factor;
	for (int i = 0;i < pol1.Coefficient.size();i++) {
		pol1.Coefficient[i] *= Fac;
	}
	return pol1;
}
std::ostream& operator<<(std::ostream& os, const Polynomial& pol) {
	if (pol.Coefficient.size() == 0) {
		os << 0;
		return os;
	}
	if (pol.Coefficient[0] != Complex(0, 0)) {
		os << pol.Coefficient[0];
	}
	for (int i = 1;i < pol.Coefficient.size();i++) {
		if (pol.Coefficient[i] == Complex(0, 0)) {
			continue;
		}
		os << "+(" << pol.Coefficient[i] << ")" << "x^" << i;
	}
	return os;
}

void Swap(Polynomial& com1, Polynomial& com2) {
	std::swap(com1.Coefficient, com2.Coefficient);

}

PolyGDC_Result GetGCD(Polynomial pol1, Polynomial pol2) {
	pol1.Sort(); pol2.Sort();

	// 边界：第二参数为零多项式时，gcd(pol1, 0) = pol1（归一化），系数 R1 = 1/首项, R2 = 0
	if (pol2.IsEmpty()) {
		if (pol1.IsEmpty()) {
			return PolyGDC_Result();   // gcd(0,0) = 0
		}
		Complex inv = Complex(1, 0) / pol1.Coefficient.back();
		pol1 *= inv;
		return PolyGDC_Result{ std::move(pol1), Polynomial(inv), Polynomial(Complex(0, 0)) };
	}
	// 边界：第一参数为零多项式时对称处理
	if (pol1.IsEmpty()) {
		Complex inv = Complex(1, 0) / pol2.Coefficient.back();
		pol2 *= inv;
		return PolyGDC_Result{ std::move(pol2), Polynomial(Complex(0, 0)), Polynomial(inv) };
	}

	// 扩展欧几里得（轮转法），维护两条不变量：
	//   r_cur  = s1*pol1 + s2*pol2
	//   r_prev = t1*pol1 + t2*pol2
	// 每轮带余除法把 (r_cur, r_prev) 推进一步，系数对同步更新，
	// 直到 r_prev 为零多项式，此时 r_cur 即最大公因式
	Polynomial r_cur = pol1, r_prev = pol2;
	Polynomial s1(Complex(1, 0)), s2;              // r_cur 的 Bézout 系数对
	Polynomial t1, t2(Complex(1, 0));              // r_prev 的 Bézout 系数对

	while (!r_prev.IsEmpty()) {
		// 保证 r_cur 次数不低于 r_prev；交换时系数对跟着一起交换，不变量不破
		if (r_cur.Coefficient.size() < r_prev.Coefficient.size()) {
			Swap(r_cur, r_prev);
			Swap(s1, t1);
			Swap(s2, t2);
		}
		PolyDivison_Result qr = r_cur / r_prev;      // 带余除法：r_cur = q*r_prev + r
		Polynomial q = std::move(qr.Quotient);
		Polynomial r = std::move(qr.Remainder);
		// 新余式 r = r_cur - q*r_prev 的系数对：
		//   u1*pol1 + u2*pol2 = (s1 - q*t1)*pol1 + (s2 - q*t2)*pol2
		Polynomial u1 = s1 - q * t1;
		Polynomial u2 = s2 - q * t2;
		// 轮转推进（注意顺序：先用旧值，再覆盖）
		r_cur = r_prev;
		r_prev = r;
		s1 = t1;
		s2 = t2;
		t1 = u1;
		t2 = u2;
	}

	// 归一化：GDC 首一化，系数对同步缩放
	Complex inv = Complex(1, 0) / r_cur.Coefficient.back();
	r_cur *= inv;
	s1 *= inv;
	s2 *= inv;
	return PolyGDC_Result{ std::move(r_cur), std::move(s1), std::move(s2) };
}

bool IsExactlyDivisible(Polynomial pol1, Polynomial pol2) {
	return (pol2 / pol1).Remainder.IsEmpty();
}
void NaiveConvolve(const PolyView& a, const PolyView& b, std::vector<Complex>& result) {
	if (a.len == 0 || b.len == 0) {
		result.clear();
		return;
	}
	result.assign(a.len + b.len - 1, Complex(0, 0));   // 一次分配到位
	for (size_t i = 0; i < a.len; i++) {
		if (a[i] == Complex(0, 0)) continue;           // 跳过零系数
		for (size_t j = 0; j < b.len; j++) {
			result[i + j] += a[i] * b[j];
		}
	}
}
Polynomial KaratsubaMul(const PolyView& a,const PolyView& b, int threshold) {
	// 长度不等时按同一个 m 拆分会错位（入口 operator* 已补齐等长，
	// 此处仅防御 sumA/sumB 经 Sort 砍零后长度不一致的边界情况）
	if (a.len != b.len) {
		std::vector<Complex> tmp;
		NaiveConvolve(a, b, tmp);
		return Polynomial(tmp);
	}
	// 基本情况：规模太小，分治的开销不划算，直接朴素算
	if (a.len <= threshold || b.len <= threshold) {
		std::vector<Complex> tmp;
		NaiveConvolve(a, b, tmp);
		return Polynomial(tmp);
	}
	size_t m = a.len / 2;
	PolyView aLow = a.LowHalf(), aHigh = a.HighHalf();
	PolyView bLow = b.LowHalf(), bHigh = b.HighHalf();

	// Karatsuba 核心：只用 3 次半长乘法
	Polynomial z0 = KaratsubaMul(aLow, bLow, threshold);
	Polynomial z2 = KaratsubaMul(aHigh, bHigh, threshold);
	// z1 = (aLow+aHigh)*(bLow+bHigh) - z0 - z2
	// 以低半段为底构造和：先把底补零到高半段长度，再把高半段逐项加上。
	// 旧写法以完整拷贝为底、push_back 后再 resize，奇数长度时会把高半段
	// 多出的最后一项砍掉、留下低半段末尾的残留系数，导致最高次项错误
	Polynomial sumA(aLow);
	Polynomial sumB(bLow);
	sumA.Coefficient.resize(aHigh.len, Complex(0, 0));
	sumB.Coefficient.resize(bHigh.len, Complex(0, 0));
	for (size_t i = 0; i < aHigh.len; i++) {
		sumA.Coefficient[i] += aHigh[i];
	}
	for (size_t i = 0; i < bHigh.len; i++) {
		sumB.Coefficient[i] += bHigh[i];
	}
	sumA.Sort();
	sumB.Sort();
	Polynomial z1 = KaratsubaMul(sumA.View(), sumB.View(), threshold);
	z1 -= z0;
	z1 -= z2;
	return z0 + (z1 >> (unsigned int)m) + (z2 >> (unsigned int)(2 * m));
}

/// <summary>原地快速傅里叶变换（迭代版 Cooley-Tukey）</summary>
/// <param name="a">要变换的数组，长度必须是 2 的幂；结果原地写回</param>
/// <param name="inverse">false = 正向变换；true = 逆变换（最后自动除以长度）</param>
void FFT(std::vector<Complex_Double>& a, bool inverse) {
	int n = (int)a.size();
	if (n <= 1) {
		return;
	}

	// ---- 第一步：位反转置换 ----
	// 递归版"按奇偶分组"拆到底后，位置 i 上应放着原下标"二进制位反转"的元素。
	// j 始终是 i 的位反转值：i 递增时，j 的变化规律是从高位往低位把 1 变 0，
	// 遇到第一个 0 再变 1（可用 i=1..7 手推验证）10000，10xxx..01000,01xxx
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		while (j & bit) {   // 找 j 从高位起的第一个 0 位
			j ^= bit;       // 该位 1 → 0
			bit >>= 1;
		}
		j ^= bit;           // 该位 0 → 1


		//这里本质上是在模拟进位操作，j相当于反向排列的二进制数，100的下一位是010，再下一位是110，以此类推（实际上，恰好与i对应）
		//顺序转化了之后，例如我们期望奇数位在前，偶数位在后，类似的递推，所以的i位的数字第log(2,n)-i重要，因此采用位反码

		if (i < j) {        // 每对只交换一次
			std::swap(a[i], a[j]);
		}
	}
	//得到位反码列，位反码列中，越靠前的元素越“偶”

	// ---- 第二步：log n 层蝴蝶 ----
	// len = 本层合并段的长度：2, 4, 8, ..., n
	for (int len = 2; len <= n; len <<= 1) {
		// 每层只算一次单位根，段内用递推（乘 wlen）推进，避免逐个算 cos/sin
		double angle = 2.0 * 3.14159265358979323846 / len * (inverse ? 1.0 : -1.0);
		Complex_Double wlen(std::cos(angle), std::sin(angle));
		int half = len >> 1;
		for (int i = 0; i < n; i += len) {     // 逐段处理
			//似乎是这样的，考虑数组，12345678
			//先弄1234，5678，由于之前的反转，这表示第一次奇偶分离，它们之间差一个wlen
			//然后12，34，56，78 它们表示第二次奇偶分离，它们之间差2分之一个wlen
			//最后1，2，3，4，5，6，7，8 它们表示第三次奇偶分离的结果，它们之间差4分之一个wlen

			Complex_Double w(1.0, 0.0);        // 段内旋转因子，从 ω⁰ = 1 起步
			for (int j = 0; j < half; j++) {   // 段内逐只蝴蝶
				Complex_Double u = a[i + j];
				Complex_Double v = a[i + j + half] * w;//它们之间的第迭代位差了个奇偶，其余差不多

				a[i + j] = u + v;              // 前半 = u + v
				a[i + j + half] = u - v;       // 后半 = u − v
				w *= wlen;                     // 递推出下一个旋转因子，j加了，相当于取多项式的下一项，自然要乘以wlen，得出下一个旋转因子
			}


		}
	}

	//本质上，这里我是希望按照“偶度”，输出一堆点

	// ---- 第三步：逆变换归一化（除以 n）----
	if (inverse) {
		for (int i = 0; i < n; i++) {
			a[i].real /= n;
			a[i].imag /= n;
		}
	}
}

/// <summary>用 FFT 计算两个多项式的乘积（内部浮点，系数较整时结果舍入后精确）</summary>
/// <param name="a">第一个多项式视图</param>
/// <param name="b">第二个多项式视图</param>
/// <returns>乘积多项式；浮点误差约 1e-12 量级，出口由 Complex 构造自动舍入</returns>
Polynomial FFTMul(const Polynomial& a, const Polynomial& b) {
	if (a.Coefficient.size() == 0 || b.Coefficient.size() == 0) {
		return Polynomial();//任一为空，乘积为零多项式
	}
	int resultLen = (int)(a.Coefficient.size() + b.Coefficient.size()) - 1;   // 乘积的项数
	int n = 1;
	while (n < resultLen) {
		n <<= 1;                                // 补零到 ≥ resultLen 的 2 的幂
	}

	// 入口：系数转成 double 复数，其余位置补 0
	std::vector<Complex_Double> fa(n, Complex_Double(0.0, 0.0));
	std::vector<Complex_Double> fb(n, Complex_Double(0.0, 0.0));
	for (size_t i = 0; i < a.Coefficient.size(); i++) {
		fa[i] = Complex_Double(a.Coefficient[i]);           // Rational → double
	}
	for (size_t i = 0; i < b.Coefficient.size(); i++) {
		fb[i] = Complex_Double(b.Coefficient[i]);
	}

	FFT(fa, false);                             // 两个正向变换
	FFT(fb, false);
	for (int i = 0; i < n; i++) {
		fa[i] *= fb[i];                         // 点值表示：逐点相乘
	}
	FFT(fa, true);                              // 逆变换还原系数

	// 出口：double → Rational（自动四舍五入到 1e-8 精度）
	std::vector<Complex> result(resultLen);
	for (int i = 0; i < resultLen; i++) {
		result[i] = Complex(fa[i].real, fa[i].imag);
	}
	return Polynomial(result);
}
int Complexity_Comparison(const Polynomial& a, const Polynomial& b) {
	if (a.Coefficient.size() != b.Coefficient.size()) {
		if (a.Coefficient.size() < b.Coefficient.size()) {
			return 1;
		}
		return -1;
	}
	for (int i = a.Coefficient.size()-1;i >= 0;i--) {
		if (a.Coefficient[i].real != b.Coefficient[i].real) {
			if (a.Coefficient[i].real < b.Coefficient[i].real) {
				return 1;
			}
			return -1;
		}
	}
	return 0;
}
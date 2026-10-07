#pragma once
#include"Polynomial.h"
#include <math.h>
class MulFunction_Node;

enum Function_Type//让简单的类型放在前面
{
	POL,//多项式（包含常数与幂函数）
	EXP,//指数（默认自然对数为底数）
	LN,//对数（默认自然对数为底数）
	SIN,//正弦
	COS,//余弦
	MUL,//函数相乘
	PUL,//函数相加
	MIN,//函数相减
	DIV,//函数相除
	COMP,//复合
};
static double PI = 3.14159265358979323846;
static double NaturalE = 2.718281828459045235360;
class Function_Node {
private:
	std::shared_ptr<Function_Node> rchild_Function;
	std::shared_ptr<Function_Node> lchild_Function;
	Polynomial valID;
	Function_Type Type;
	int depth;
	Complex GetConstant() const {   // 取出常数值（限 IsConstant 为真时调用）
		return valID.IsEmpty() ? Complex(0, 0) : valID.Coefficient[0];
	}
	void Sort();
	bool IsNeedSwap();//在构造时调用,表示是否需要交换左右孩子的位置
public:
	Function_Node(Function_Type type,const Function_Node& lch,const Function_Node& rch) :
		Type(type){
		if (type == SIN || type == COS || type == EXP || type == LN || type==POL) {
			throw std::runtime_error("Basic Functions needn't ex resource");
		}
		lchild_Function = std::make_shared<Function_Node>(lch);
		rchild_Function = std::make_shared<Function_Node>(rch);
		depth = lchild_Function->depth > rchild_Function->depth ? lchild_Function->depth + 1 : rchild_Function->depth + 1;
		Sort();
	};
	Function_Node(Function_Type type, std::shared_ptr<Function_Node> lch, std::shared_ptr<Function_Node> rch) :
		Type(type),
		lchild_Function(lch),
		rchild_Function(rch)
	{
		if (type == SIN || type == COS || type == EXP || type == LN || type == POL) {
			throw std::runtime_error("Basic Functions needn't ex resource");
		}
		depth = lchild_Function->depth > rchild_Function->depth ? lchild_Function->depth + 1 : rchild_Function->depth + 1;
		Sort();
	};
	Function_Node(Function_Type type):
		Type(type),
		lchild_Function(nullptr),
		rchild_Function(nullptr)
	{
		if (type != SIN && type != COS && type != EXP && type != LN) {
			throw std::runtime_error("Pol or Complex type's Function must construct with more parameter");
		}
		depth = 0;
		Sort();
	}
	Function_Node(Polynomial valid) ://不输入别的，默认为多项式类型
		valID(valid),
		Type(POL),
		rchild_Function(nullptr),
		lchild_Function(nullptr)
	{
		depth = 0;
		Sort();
	};
	Function_Node(double val=0) ://输入常数,默认返回0
		valID(Polynomial(Complex(Rational(val)))),
		Type(POL),
		rchild_Function(nullptr),
		lchild_Function(nullptr)
	{
		depth = 0;
		Sort();
	};
	double CalculatedValue(double val);
	Function_Node Differentiation();
	Function_Node operator()(const Function_Node& funcin);
	MulFunction_Node operator()(const MulFunction_Node& funcin);
	double operator()(double val);

	//化简判断
	friend bool IsEqul(const Function_Node& f,const Function_Node& g);
	bool IsZero()const { return Type == POL && valID.IsEmpty(); }
	bool IsOne()const { return Type == POL && valID.Coefficient.size() == 1 && valID.Coefficient[0] == Complex(1); }
	bool IsConstant() const {   // 是否为常数（含零多项式）
		return Type == POL && valID.Coefficient.size() <= 1;
	}
	bool IsIdentity() const {   // 是否为 f(x) = x
		return Type == POL && valID.Coefficient.size() == 2
			&& valID.Coefficient[0] == Complex(0, 0)
			&& valID.Coefficient[1] == Complex(1, 0);
	}
};
Function_Node operator+(const Function_Node& func1, const Function_Node& func2);
Function_Node operator-(const Function_Node& func1, const Function_Node& func2);
Function_Node operator-(const Function_Node& func);
Function_Node operator*(const Function_Node& func1, const Function_Node& func2);
Function_Node operator/(const Function_Node& func1, const Function_Node& func2);
bool IsEqul(const Function_Node& f, const Function_Node& g);
enum MulFunction_Type
{
	ORD,//只读取第一位的值，做一元运算

	ORDPUL,//对每个变量的一元函数求和
	ORDMUL,//对每个变量的一元函数求积
	ORDCOMP,//对得到的结果进行一元的复合运算

	MULCOMP,//对得到的结果进行多变量的复合运算
	MULPUL,//对多个复合函数的结果相加
	MULMUL//对多个复合函数得到的结果相乘

};
struct MUL_Simpily_Data//给出退化时化简要用到的数据
{
	bool IsNeedSimply;
	int NewORDIndex;
};
class MulFunction_Node {
private:
	std::vector< std::shared_ptr < MulFunction_Node > > MULSon;
	std::shared_ptr< MulFunction_Node >  MULMain;
	std::vector< std::shared_ptr<Function_Node> > ORDSon;
	std::shared_ptr <Function_Node> ORDMain;
	MulFunction_Type Type;
	int ORD_Index=0;
	bool IsZero() const;
	bool IsOne() const;
	bool IsConstant() const;
	MUL_Simpily_Data IsNeedDegeneration() const;
	/// @brief 尝试把本节点精确折叠成一个常数值（供 Sort 第 2 步塌缩使用）
	/// @param out 成功时写入折叠出的常数节点（Function_Node 的 POL 类型）
	/// @return 能精确折叠则返回 true；含非常数分量或复合外层为超越函数等"有损"情形返回 false
	bool TryGetConstantValue(Function_Node& out) const;

	void Sort();
public:
	//构造平凡的复合函数
	//1，从数组构建
	MulFunction_Node(MulFunction_Type type,const std::vector<MulFunction_Node>& functions) {
		Type = type;
		if (type != MULPUL&&type!=MULMUL) {
			throw std::runtime_error("mismatched type");
		}
		MULSon.resize(functions.size());
		for (int i = 0;i < functions.size();i++) {
			MULSon[i] = std::make_shared<MulFunction_Node>(functions[i]);
		}
		Sort();
	};

	MulFunction_Node(MulFunction_Type type,const std::vector< std::shared_ptr < MulFunction_Node > >& functions) {
		Type = type;
		if (type != MULPUL && type != MULMUL) {
			throw std::runtime_error("mismatched type");
		}
		MULSon=functions;
		Sort();
	};
	//2，从两个一元函数构建
	MulFunction_Node(MulFunction_Type type,const Function_Node& function1,const Function_Node& function2) {
		Type = type;
		if (type != ORDPUL && type != ORDMUL) {
			throw std::runtime_error("mismatched type");
		}
		ORDSon.push_back(std::make_shared<Function_Node>(function1));
		ORDSon.push_back(std::make_shared<Function_Node>(function2));
		Sort();
	};

	//构造复合复合函数
	//1.从数组构造
	MulFunction_Node(MulFunction_Type type,const std::vector<Function_Node>& functions) {
		Type = type;
		if (type != ORDPUL && type != ORDMUL) {
			throw std::runtime_error("mismatched type");
		}
		ORDSon.resize(functions.size());
		for (int i = 0;i < functions.size();i++) {
			ORDSon[i] = std::make_shared<Function_Node>(functions[i]);
		}
		Sort();
	};
	MulFunction_Node(MulFunction_Type type,const std::vector<std::shared_ptr <Function_Node>>& functions) {
		Type = type;
		if (type != ORDPUL && type != ORDMUL) {
			throw std::runtime_error("mismatched type");
		}
		ORDSon = functions;
		Sort();
	};
	//2.从两个复合函数直接构造
	MulFunction_Node(MulFunction_Type type,const MulFunction_Node& function1,const MulFunction_Node& function2) {
		Type = type;
		if (type != MULPUL && type != MULMUL) {
			throw std::runtime_error("mismatched type");
		}
		MULSon.push_back(std::make_shared<MulFunction_Node>(function1));
		MULSon.push_back(std::make_shared<MulFunction_Node>(function2));
		Sort();
	};

	//构造复合函数，外一元内多元
	MulFunction_Node(const MulFunction_Node& mul,const Function_Node& ord) {
		Type = ORDCOMP;
		MULMain = std::make_shared<MulFunction_Node>(mul);
		ORDMain = std::make_shared<Function_Node>(ord);
		Sort();
	}
	//构造复合函数，外多元内多元
	MulFunction_Node(const MulFunction_Node& main,const std::vector<MulFunction_Node>& functions) {
		Type = MULCOMP;
		MULMain = std::make_shared<MulFunction_Node>(main);
		MULSon.resize(functions.size());
		for (int i = 0;i < functions.size();i++) {
			MULSon[i] = std::make_shared<MulFunction_Node>(functions[i]);
		}
		Sort();
	}

	MulFunction_Node(const MulFunction_Node& main,const std::vector< std::shared_ptr < MulFunction_Node > >& functions) {
		Type = MULCOMP;
		MULMain = std::make_shared<MulFunction_Node>(main);
		MULSon=functions;
		Sort();
	}

	//通过一个1元函数，直接生成一个函数
	MulFunction_Node(const Function_Node& func,int index=0) {
		Type = ORD;
		ORD_Index = index;
		ORDMain = std::make_shared<Function_Node>(func);
		Sort();
	}
	double CalculatedValue(std::vector<double> val);//只读自己关心的几位
	MulFunction_Node Differentiation(int index);
	MulFunction_Node operator()(std::vector<MulFunction_Node>& funcin);
	MulFunction_Node operator()(std::vector< std::shared_ptr < MulFunction_Node > >& funcin);
	double operator()(std::vector<double> val);

	friend MulFunction_Node operator+(const MulFunction_Node& func1, const MulFunction_Node& func2);
	friend MulFunction_Node operator-(const MulFunction_Node& func1, const MulFunction_Node& func2);
	friend MulFunction_Node& operator+=(MulFunction_Node& func1, const MulFunction_Node& func2);
	friend MulFunction_Node& operator-=(MulFunction_Node& func1, const MulFunction_Node& func2);
	friend MulFunction_Node operator-(const MulFunction_Node& func);
	friend MulFunction_Node operator*(const MulFunction_Node& func1, const MulFunction_Node& func2);
	friend MulFunction_Node operator/(const MulFunction_Node& func1, const MulFunction_Node& func2);
	friend MulFunction_Node& operator*=(MulFunction_Node& func1, const MulFunction_Node& func2);
	friend MulFunction_Node& operator/=(MulFunction_Node& func1, const MulFunction_Node& func2);
};
MulFunction_Node operator+(const MulFunction_Node& func1, const MulFunction_Node& func2);
MulFunction_Node operator-(const MulFunction_Node& func1, const MulFunction_Node& func2);

MulFunction_Node& operator+=(MulFunction_Node& func1, const MulFunction_Node& func2);
MulFunction_Node& operator-=(MulFunction_Node& func1, const MulFunction_Node& func2);

MulFunction_Node operator-(const MulFunction_Node& func);
MulFunction_Node operator*(const MulFunction_Node& func1, const MulFunction_Node& func2);
MulFunction_Node operator/(const MulFunction_Node& func1, const MulFunction_Node& func2);

MulFunction_Node& operator*=(MulFunction_Node& func1, const MulFunction_Node& func2);
MulFunction_Node& operator/=(MulFunction_Node& func1, const MulFunction_Node& func2);


inline static Function_Node NEG(Polynomial({ Complex(0),Complex(-1) }));
inline static Function_Node CONST_1(1);
inline static Function_Node CONST_2(2);
inline static Function_Node CONST_X(Polynomial({ Complex(0),Complex(1) }));
inline static Function_Node CONST_0(0);
inline static Function_Node REC(DIV, CONST_1, CONST_X);

static struct FOO
{
	std::shared_ptr<Function_Node> fooNEG = std::make_shared<Function_Node>(NEG);
	std::shared_ptr<Function_Node> fooREC = std::make_shared<Function_Node>(REC);
	std::shared_ptr<Function_Node> fooCONST_1 = std::make_shared<Function_Node>(CONST_1);
	std::shared_ptr<Function_Node> fooCONST_2 = std::make_shared<Function_Node>(CONST_2);
	std::shared_ptr<Function_Node> fooCONST_0 = std::make_shared<Function_Node>(CONST_0);
	std::shared_ptr<Function_Node> fooCONST_X = std::make_shared<Function_Node>(CONST_X);
} foo;

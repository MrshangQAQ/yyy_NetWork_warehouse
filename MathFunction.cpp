#include"MathFunction.h"
double Function_Node::CalculatedValue(double val) {
	switch (Type)
	{
	case MUL:
		return lchild_Function->CalculatedValue(val) * rchild_Function->CalculatedValue(val);
	case MIN:
		return lchild_Function->CalculatedValue(val) - rchild_Function->CalculatedValue(val);
	case PUL:
		return lchild_Function->CalculatedValue(val) + rchild_Function->CalculatedValue(val);
	case DIV:
		return lchild_Function->CalculatedValue(val) / rchild_Function->CalculatedValue(val);
	case COMP:
		return lchild_Function->CalculatedValue(rchild_Function->CalculatedValue(val));
	case SIN:
		return std::sin(val);
	case COS:
		return std::cos(val);
	case EXP:
		return std::exp(val);
	case LN:
		return std::log(val);
	case POL:
		return valID.CalculatedValue(Complex_Double(val,0)).real;
	default:
		throw std::runtime_error("Unknow type");
	}
}
Function_Node Function_Node::Differentiation() {
	switch (Type)
	{
	case MUL:
		return (rchild_Function->Differentiation() * *lchild_Function)+(lchild_Function->Differentiation()* *rchild_Function);
	case MIN:
		return lchild_Function->Differentiation() - rchild_Function->Differentiation();
	case PUL:
		return lchild_Function->Differentiation() + rchild_Function->Differentiation();
	case DIV:
		return ((lchild_Function->Differentiation() * *rchild_Function)-(rchild_Function->Differentiation() * *lchild_Function))/(*rchild_Function * *rchild_Function);
	case COMP:
		return rchild_Function->Differentiation()*((lchild_Function->Differentiation())(*rchild_Function));
	case SIN:
		return Function_Node(COS);
	case COS:
		return -Function_Node(SIN);
	case EXP:
		return Function_Node(EXP);
	case LN:
		return REC;
	case POL:
		return Function_Node(GetDifferentiation(valID));
	default:
		throw std::runtime_error("Unknow type");
	}
}
bool Function_Node::IsNeedSwap() {
	if (lchild_Function->Type > rchild_Function->Type) {
		return true;
	}
	if (lchild_Function->Type == rchild_Function->Type) {
		if (lchild_Function->Type == POL) {
			if (Complexity_Comparison(lchild_Function->valID, rchild_Function->valID)==-1) {
				return true;
			}
			return false;
		}
		return lchild_Function->depth > rchild_Function->depth;//如果是初等函数节点，也会返回false
	}
	return false;
}
void Function_Node::Sort() {
	// 叶子节点（含多项式）没有孩子，直接返回
	if (lchild_Function == nullptr || rchild_Function == nullptr) {
		return;
	}
	if ((Type == MUL || Type == PUL)&&IsNeedSwap()) {
		std::swap(lchild_Function,rchild_Function);
	}
	// 第一步：0、1、恒等函数的快速化简
	if (Type == PUL) {
		if (lchild_Function->IsZero()) { *this = Function_Node(*rchild_Function); return; } // 0+f=f
		if (rchild_Function->IsZero()) { *this = Function_Node(*lchild_Function); return; } // f+0=f
		if (IsEqul(*lchild_Function, *rchild_Function)) { *this = Function_Node(MUL, foo.fooCONST_2, lchild_Function);return; }
	}
	else if (Type == MIN) {
		if (rchild_Function->IsZero()) { *this = Function_Node(*lchild_Function); return; } // f-0=f
		if (lchild_Function->IsZero()) { *this = -(*rchild_Function); return; }             // 0-f=-f（不是 f！）
		if (IsEqul(*lchild_Function, *rchild_Function)) { *this = Function_Node();return; }
	}
	else if (Type == MUL) {
		if (lchild_Function->IsZero() || rchild_Function->IsZero()) { *this = Function_Node(0); return; }
		if (lchild_Function->IsOne()) { *this = Function_Node(*rchild_Function); return; }  // 1*f=f
		if (rchild_Function->IsOne()) { *this = Function_Node(*lchild_Function); return; }  // f*1=f
	}
	else if (Type == DIV) {
		if (rchild_Function->IsZero()) { return; }   // f/0 绝不能化简，保持原样
		if (rchild_Function->IsOne()) { *this = Function_Node(*lchild_Function); return; }  // f/1=f
		if (lchild_Function->IsZero()) { *this = Function_Node(0); return; }                // 0/f=0
		if (IsEqul(*lchild_Function, *rchild_Function)) { *this = Function_Node(1);return; }
	}
	else if (Type == COMP) {
		if (lchild_Function->IsIdentity()) { *this = Function_Node(*rchild_Function); return; } // x(g)=g
		if (rchild_Function->IsIdentity()) { *this = Function_Node(*lchild_Function); return; } // f(x)=f
		if (lchild_Function->IsConstant()) { *this = Function_Node(*lchild_Function); return; } // c(g)=c
		if (rchild_Function->IsConstant() && lchild_Function->Type == POL) {
			// 外层多项式在内层常数点处精确求值：f(c) 仍是有理数，可折叠
			Complex value = lchild_Function->valID.CalculatedValue(rchild_Function->GetConstant());
			*this = Function_Node(Polynomial(value));
			return;
		}
	}
	else {
		return;   // 其他类型不处理
	}

	// 第二步：左右孩子都是多项式 → 合并成一个多项式节点
	if (lchild_Function->Type == POL && rchild_Function->Type == POL) {
		if (Type == PUL) {
			*this = Function_Node(lchild_Function->valID + rchild_Function->valID);
		}
		else if (Type == MIN) {
			*this = Function_Node(lchild_Function->valID - rchild_Function->valID);
		}
		else if (Type == MUL) {
			*this = Function_Node(lchild_Function->valID * rchild_Function->valID);
		}
		else if (Type == DIV && rchild_Function->IsConstant()) {
			// 除数是"非零常数"才可合并（零的情形已在上面提前返回）
			*this = Function_Node(lchild_Function->valID / rchild_Function->GetConstant());
		}
	}
}
Function_Node Function_Node::operator()(const Function_Node& funcin) {
	return Function_Node(COMP, *this, funcin);
}
MulFunction_Node Function_Node::operator()(const MulFunction_Node& funcin) {
	return MulFunction_Node(funcin,*this);
}
double Function_Node::operator()(double val) {
	return CalculatedValue(val);
}
Function_Node operator+(const Function_Node& func1, const Function_Node& func2) {
	return Function_Node(PUL, func1, func2);
}
Function_Node operator-(const Function_Node& func1, const Function_Node& func2) {
	return Function_Node(MIN, func1, func2);
}
Function_Node operator-(const Function_Node& func) {
	Function_Node func2(-1);
	return Function_Node(MUL,func, func2);
}
Function_Node operator*(const Function_Node& func1, const Function_Node& func2) {
	return Function_Node(MUL, func1, func2);
}
Function_Node operator/(const Function_Node& func1, const Function_Node& func2) {
	return Function_Node(DIV, func1, func2);
}
bool IsEqul(const Function_Node& f, const Function_Node& g) {
	if (f.Type != g.Type||f.depth!=g.depth) {
		return false;
	}
	switch (f.Type)
	{
	case POL:
		return Complexity_Comparison(f.valID, g.valID) == 0;
	case EXP:
		return true;
	case LN:
		return true;
	case SIN:
		return true;
	case COS:
		return true;
	default:
		return IsEqul(*f.lchild_Function, *g.lchild_Function) && IsEqul(*f.rchild_Function, *g.rchild_Function);
	}
}


bool MulFunction_Node::IsZero() const {
	if (Type == ORD || Type == ORDCOMP) return ORDMain->IsZero();
	if (Type == MULCOMP)               return MULMain->IsZero();
	if (Type == ORDPUL || Type == MULPUL) {   // 和：全部位都是零才是零
		if (Type == ORDPUL) {
			for (int i = 0;i < ORDSon.size();i++) {
				if (!ORDSon[i]->IsZero()) {
					return false;
				}
			}
			return true;
		}
		for (int i = 0;i < MULSon.size();i++) {
			if (!MULSon[i]->IsZero()) {
				return false;
			}
		}
		return true;
	}
	if (Type == ORDMUL || Type == MULMUL) {   // 积：有一个零就是零
		if (Type == ORDMUL) {
			for (int i = 0;i < ORDSon.size();i++) {
				if (ORDSon[i]->IsZero()) {
					return true;
				}
			}
			return false;
		}
		for (int i = 0;i < MULSon.size();i++) {
			if (MULSon[i]->IsZero()) {
				return true;
			}
		}
		return false;
	}
	return false;
}
bool MulFunction_Node::IsOne() const {
	if (Type == ORD) { return ORDMain->IsOne(); }
	if (Type == MULCOMP) { return MULMain->IsOne(); }
	if (Type == ORDCOMP) { return ORDMain->IsOne(); }
	if (Type == ORDMUL) {
		for (int i = 0;i < ORDSon.size();i++) {
			if (!ORDSon[i]->IsOne()) {
				return false;
			}
		}
		return true;
	}
	if (Type == MULMUL) {
		for (int i = 0;i < MULSon.size();i++) {
			if (!MULSon[i]->IsOne()) {
				return false;
			}
		}
		return true;
	}
	return false;
}
bool MulFunction_Node::IsConstant() const {
	if (Type == ORD) { return ORDMain->IsConstant(); }
	if (Type == ORDMUL || Type == ORDPUL ){
		for (int i = 0;i < ORDSon.size();i++) {
			if (!ORDSon[i]->IsConstant()) {
				return false;
			}
		}
		return true;
	}
	if (Type == MULMUL || Type == MULPUL) {
		for (int i = 0;i < MULSon.size();i++) {
			if (!MULSon[i]->IsConstant()) {
				return false;
			}
		}
		return true;
	}
	if (Type == ORDCOMP) {
		return ORDMain->IsConstant() || MULMain->IsConstant();
	}
	if (Type == MULCOMP) {
		if (MULMain->IsConstant()) {
			return true;
		}
		for (int i = 0;i < MULSon.size();i++) {
			if (!MULSon[i]->IsConstant()) {
				return false;
			}
		}
		return true;
	}
	return false;
}
MUL_Simpily_Data MulFunction_Node::IsNeedDegeneration() const {
	if (Type == ORDMUL||Type==ORDPUL) {//平凡加法或乘法退化为ORD
		int NewIndex = -1;
		int tmp=0;//记录有多少个非中性点
		for (int i = 0;i < ORDSon.size();i++) {
			if (!ORDSon[i]->IsConstant()) {
				tmp++;
				NewIndex = i;
			}
			if (tmp >= 2) {
				return{ false,-1 };
			}
		}
		return { true,NewIndex };
	}
	if (Type == ORDCOMP) {//平凡复合退化为步需要复合
		if (ORDMain->IsIdentity()) {
			return { true,-1 };
		}
	}
	if (Type == MULMUL || Type == MULPUL) {//多元加法或多元乘法退化为平凡加法或者平凡乘法
		for (int i = 0;i < MULSon.size();i++) {
			if (!(MULSon[i]->Type == ORD && MULSon[i]->ORD_Index == i)) {
				return { false,-1 };
			}
		}
		return { true,-1 };
	}
	if (Type == MULCOMP) {//多元复合退化成为多元加法或多元乘法
		if (MULMain->Type == ORDMUL || MULMain->Type == ORDPUL) {
			// 数量对齐检查：外层分位函数的个数必须不少于内层函数个数，
			// 否则求值时 min(val.size(), ORDSon.size()) 会截断求和/求积项，退化后与原式不等价
			if (MULMain->ORDSon.size() < MULSon.size()) return { false,-1 };
			for (int i = 0;i < MULMain->ORDSon.size();i++) {
				if (!(MULMain->ORDSon[i]->IsIdentity())) {//若其所有子节点都是平凡节点且这些平凡节点钧为Identity,退化为多元加法或者多元乘法
					return {false,-1};
				}
			}
			return { true,-1 };
		}
	}
	return { false,-1 };
}
bool MulFunction_Node::TryGetConstantValue(Function_Node& out) const {
	// 一元节点（ORD）：孩子就是一个一元函数，它本身是常数才算常数
	if (Type == ORD) {
		if (!ORDMain->IsConstant()) {
			return false;
		}
		out = *ORDMain;
		return true;
	}
	// 一元和/积：孩子是逐位的一元函数，全部为常数才可精确合并。
	// 以中性元起步（和用 0、积用 1），空数组也能自然得到正确结果（空和 = 0、空积 = 1）
	if (Type == ORDPUL || Type == ORDMUL) {
		Function_Node g(Type == ORDPUL ? 0.0 : 1.0);
		for (int i = 0; i < (int)ORDSon.size(); i++) {
			if (!ORDSon[i]->IsConstant()) {
				return false;
			}
			g = (Type == ORDPUL) ? g + *ORDSon[i] : g * *ORDSon[i];
		}
		out = g;
		return true;
	}
	// 多元和/积：孩子是完整的多元函数，递归求每个孩子的常数值再合并
	if (Type == MULPUL || Type == MULMUL) {
		Function_Node g(Type == MULPUL ? 0.0 : 1.0);
		for (int i = 0; i < (int)MULSon.size(); i++) {
			Function_Node sub;
			if (!MULSon[i]->TryGetConstantValue(sub)) {
				return false;
			}
			g = (Type == MULPUL) ? g + sub : g * sub;
		}
		out = g;
		return true;
	}
	// 复合型（ORDCOMP/MULCOMP）外层可能含超越函数（如 sin），我们这里只检查外层是否为常数
	if (Type == MULCOMP || Type == ORDCOMP) {
		if (Type == ORDCOMP) {
			if (ORDMain->IsConstant()) {
				out = *ORDMain;
				return true;
			}
			out = 0;
			return false;
		}
		if (MULMain->TryGetConstantValue(out)) {
			return true;
		}
	}
	// 对常数求值会产生无理数、有理数无法精确表示，故一律不在此塌缩，保持原状交给退化步骤。
	return false;
}
void MulFunction_Node::Sort() {
	// 约定：本函数只处理"自己这一层"。子节点在各自构造时已经化简过（构造即规范），
	// 所以这里不递归进子树；命中规则后用 *this = MulFunction_Node(...) 重建自己并立即
	// return，由新对象的构造函数继续化简（链式接力），因此内部不写循环。
	// 每条规则都必须让结构严格变小（类型降级或子节点减少），否则构造里的 Sort 会无限递归。

	if (Type == ORD) {
		// 常数与位次无关：统一归到第 0 位，便于 MUL 族做位次对齐判断
		if (ORD_Index != 0 && ORDMain->IsConstant()) {
			ORD_Index = 0;
		}
		return;
	}

	// ===== 第 2 步：塌缩平凡值 =====
	// 2a. 恒零优先：零是"和"的中性元、也是"积"的吸收元，还顺带覆盖了空和/空积等边界。
	//     必须放在常数判断之前——MULMUL{0, x} 中 x 非常数，IsConstant() 判不出来，只能靠 IsZero()。
	if (IsZero()) {
		*this = MulFunction_Node(Function_Node(0.0));
		return;
	}
	// 2b. 全常数：精确求值后塌缩成"那个常数值"（如 MULPUL{2,3} → 常数 5），而非原地返回。
	//     只对 ORDPUL/ORDMUL/MULPUL/MULMUL 做；复合型（ORDCOMP/MULCOMP）外层含超越函数时
	//     （如 sin(2)）结果是无理数、有理数无法精确表示，塌缩有损，故保持原状交给退化步骤。
	Function_Node value;
	if (TryGetConstantValue(value)) {
		*this = MulFunction_Node(value);
		return;
	}

	if (Type == ORDPUL || Type == ORDMUL) {

		//1，简单剔除特殊情况
		if (ORDSon.empty()) {   // 空和 = 0、空积 = 1
			*this = MulFunction_Node(Function_Node(Type == ORDPUL ? 0.0 : 1.0));
			return;
		}
		//2，开始进行退化整理

		MUL_Simpily_Data data = IsNeedDegeneration();

		if (!data.IsNeedSimply) {
			return;
		}
		if (data.NewORDIndex == -1) {
			// 所有位都是常数（-1 表示不需要位次）：合并成一个常数节点
			Function_Node g = *ORDSon[0];
			for (int i = 1; i < (int)ORDSon.size(); i++) {
				g = (Type == ORDPUL) ? g + *ORDSon[i] : g * *ORDSon[i];
			}
			*this = MulFunction_Node(g);
			return;
		}
		// 只有第 p 位不是常数：把其余常数并入该位，整体退化为 ORD_p(g)
		int p = data.NewORDIndex;
		Function_Node g = *ORDSon[p];
		for (int i = 0; i < (int)ORDSon.size(); i++) {
			if (i == p) {
				continue;
			}
			g = (Type == ORDPUL) ? g + *ORDSon[i] : g * *ORDSon[i];
		}
		*this = MulFunction_Node(g, p);
		return;
	}

	if (Type == MULPUL || Type == MULMUL) {
		if (MULSon.empty()) {   // 空和 = 0、空积 = 1
			*this = MulFunction_Node(Function_Node(Type == MULPUL ? 0.0 : 1.0));
			return;
		}
		if (MULSon.size() == 1) {   // Σ{f} = f、Π{f} = f
			*this = *MULSon[0];
			return;
		}
		// 剔除中性元：求和剔恒零项、求积剔恒一项
		std::vector<std::shared_ptr<MulFunction_Node>> kept;
		for (int i = 0; i < (int)MULSon.size(); i++) {
			bool isNeutral = (Type == MULPUL) ? MULSon[i]->IsZero() : MULSon[i]->IsOne();
			if (!isNeutral) {
				kept.push_back(MULSon[i]);
			}
		}
		if (kept.size() != MULSon.size()) {   // 确实剔掉了东西才重建，避免原地打转
			if (kept.empty()) {
				*this = MulFunction_Node(Function_Node(Type == MULPUL ? 0.0 : 1.0));
				return;
			}
			if (kept.size() == 1) {
				*this = *kept[0];
				return;
			}
			*this = MulFunction_Node(Type, kept);
			return;
		}
		// 位次对齐降级：所有子项都是"只关注第 i 位"的 ORD_i → 降级为 ORDPUL / ORDMUL
		if (IsNeedDegeneration().IsNeedSimply) {
			std::vector<std::shared_ptr<Function_Node>> newSon;
			for (int i = 0; i < (int)MULSon.size(); i++) {
				newSon.push_back(MULSon[i]->ORDMain);
			}
			*this = MulFunction_Node(Type == MULPUL ? ORDPUL : ORDMUL, newSon);
			return;
		}
		return;
	}

	if (Type == MULCOMP) {
		if (MULMain->IsOne()) {    // 外层恒一 → 整体恒一
			*this = MulFunction_Node(Function_Node(1.0));
			return;
		}
		// 外层是"每个分量都恒等"的按位求和/求积（即 Main(u) = u0+u1+... 或 u0*u1*...）时，
		// 复合就退化为对子项直接求和/求积
		if (IsNeedDegeneration().IsNeedSimply) {
			if (MULMain->Type == ORDPUL) {
				*this = MulFunction_Node(MULPUL, MULSon);
			}
			else {
				*this = MulFunction_Node(MULMUL, MULSon);
			}
			return;
		}
		return;
	}

	if (Type == ORDCOMP && ORDMain->IsIdentity()) { *this = *MULMain; return; }                    // 外层恒等：x(g) = g
	if (Type == ORDCOMP && ORDMain->IsConstant()) { *this = MulFunction_Node(*ORDMain); return; }  // 外层常数：c(g) = c
}
double MulFunction_Node::CalculatedValue(std::vector<double> val) {
	if (Type == ORD) {
		return ORDMain->CalculatedValue(val[ORD_Index]);
	}
	if (Type == ORDPUL) {
		double sum = 0;
		int nums = val.size() < ORDSon.size() ? val.size() : ORDSon.size();
		for (int i = 0;i < nums;i++) {
			sum += ORDSon[i]->CalculatedValue(val[i]);
		}
		return sum;
	}
	if (Type == ORDMUL) {
		double mul = 1;
		int nums = val.size() < ORDSon.size() ? val.size() : ORDSon.size();
		for (int i = 0;i < nums;i++) {
			mul *= ORDSon[i]->CalculatedValue(val[i]);
		}
		return mul;
	}
	if (Type == ORDCOMP) {
		return ORDMain->CalculatedValue(MULMain->CalculatedValue(val));
	}
	if (Type == MULPUL) {
		double sum = 0;
		for (int i = 0;i < MULSon.size();i++) {
			sum += MULSon[i]->CalculatedValue(val);
		}
		return sum;
	}
	if (Type == MULMUL) {
		double mul = 1;
		for (int i = 0;i < MULSon.size();i++) {
			mul *= MULSon[i]->CalculatedValue(val);
		}
		return mul;
	}
	if (Type == MULCOMP) {
		std::vector<double> newval;
		newval.resize(MULSon.size(), 0);
		for (int i = 0;i < newval.size();i++) {
			newval[i] = MULSon[i]->CalculatedValue(val);
		}
		return MULMain->CalculatedValue(newval);
	}
	throw std::runtime_error("unknow function type");
}
MulFunction_Node MulFunction_Node::Differentiation(int index) {
	if (IsConstant()) {
		return MulFunction_Node(Function_Node(0));
	}
	if (Type == ORD) {
		if (index == ORD_Index) {
			return MulFunction_Node(ORDMain->Differentiation(),ORD_Index);
		}
		return Function_Node(0);
	}
	if (Type == ORDPUL) {
		// 越界保护：index 指向不存在的变量时，偏导恒为常数 0
		if (index < 0 || index >= (int)ORDSon.size()) {
			return MulFunction_Node(0);
		}
		// 保持位次：ORDPUL 的第 i 个子函数固定对应第 i 个变量，
		// 仅第 index 位换成导数、其余位置置零（避免偏导被 ORD 误读为第一位变量）
		std::vector<std::shared_ptr<Function_Node>> newdata;
		newdata.resize(ORDSon.size());
		for (int i = 0; i < (int)ORDSon.size(); i++) {
			if (i == index) {
				newdata[i] = std::make_shared<Function_Node>(ORDSon[i]->Differentiation());
			}
			else {
				newdata[i] = std::make_shared<Function_Node>(0);
			}
		}
		return MulFunction_Node(ORDPUL, newdata);
	}
	if (Type == ORDMUL) {
		// 越界保护：index 指向不存在的变量时，偏导恒为常数 0
		if (index < 0 || index >= (int)ORDSon.size()) {
			return MulFunction_Node(0);
		}
		std::vector<std::shared_ptr<Function_Node>> newdata = ORDSon;
		newdata[index] =std::make_shared<Function_Node>( ORDSon[index]->Differentiation());
		return MulFunction_Node(ORDMUL, newdata);
	}
	if (Type == ORDCOMP) {
		return (ORDMain->Differentiation())(*MULMain) * (MULMain->Differentiation(index));
	}
	if (Type == MULMUL) {
		MulFunction_Node sum(0);
		for (int i = 0;i < MULSon.size();i++) {
			MulFunction_Node mul(1);
			for (int j = 0;j < MULSon.size();j++) {
				if (j != i) {
					mul *= *MULSon[j];
				}
				else {
					mul *= MULSon[j]->Differentiation(index);
				}
			}
			sum += mul;
		}
		return sum;
	}
	if (Type == MULPUL) {
		MulFunction_Node sum(0);
		for (int i = 0;i < MULSon.size();i++) {
			sum += MULSon[i]->Differentiation(index);
		}
		return sum;
	}
	if (Type == MULCOMP) {
		MulFunction_Node sum(0);
		for (int i = 0;i < MULSon.size();i++) {
			sum += (MULMain->Differentiation(i))(MULSon) * MULSon[i]->Differentiation(index);
		}
		return sum;
	}
	throw std::runtime_error("unknow function type");
}

MulFunction_Node MulFunction_Node::operator()(std::vector<MulFunction_Node>& funcin) {
	return MulFunction_Node(*this, funcin);
}
MulFunction_Node MulFunction_Node::operator()(std::vector< std::shared_ptr < MulFunction_Node > >& funcin) {
	return MulFunction_Node(*this, funcin);
}
double MulFunction_Node::operator()(std::vector<double> val) {
	return CalculatedValue(val);
}
MulFunction_Node operator+(const MulFunction_Node& func1, const MulFunction_Node& func2) {
	if (func1.Type == MULPUL) {
		std::vector<std::shared_ptr<MulFunction_Node>> sons = func2.MULSon;
		sons.push_back(std::make_shared<MulFunction_Node>(func1));
		return MulFunction_Node(MULPUL, sons);   // 走构造函数重建，由构造函数自动化简
	}
	if(func2.Type==MULPUL){
		std::vector<std::shared_ptr<MulFunction_Node>> sons = func1.MULSon;
		sons.push_back(std::make_shared<MulFunction_Node>(func2));
		return MulFunction_Node(MULPUL, sons);   // 走构造函数重建，由构造函数自动化简
	}
	return MulFunction_Node(MULPUL,func1,func2);
}
MulFunction_Node operator-(const MulFunction_Node& func) {
	return NEG(func);
}
MulFunction_Node operator-(const MulFunction_Node& func1, const MulFunction_Node& func2) {
	if (func1.Type == MULPUL) {
		MulFunction_Node newfunc1 = func1;
		newfunc1.MULSon.push_back(std::make_shared<MulFunction_Node>(-func2));
		newfunc1.Sort();
		return newfunc1;
	}
	return func1+(-func2);
}
MulFunction_Node& operator+=(MulFunction_Node& func1, const MulFunction_Node& func2) {
	if (func1.Type == MULPUL) {
		func1.MULSon.push_back(std::make_shared<MulFunction_Node>(func2));
		func1.Sort();
		return func1;
	}
	MulFunction_Node cope = func1;
	func1 = MulFunction_Node(MULPUL,cope, func2);
	return func1;
}

MulFunction_Node& operator-=(MulFunction_Node& func1, const MulFunction_Node& func2) {
	if (func1.Type == MULPUL) {
		func1.MULSon.push_back(std::make_shared<MulFunction_Node>(-func2));
		func1.Sort();
		return func1;
	}
	MulFunction_Node cope = func1;
	MulFunction_Node NEGFUNC2 = -func2;
	func1 = MulFunction_Node(MULPUL, cope, NEGFUNC2);
	return func1;
}

MulFunction_Node operator*(const MulFunction_Node& func1, const MulFunction_Node& func2) {
	if (func1.Type == MULMUL) {
		std::vector<std::shared_ptr<MulFunction_Node>> sons = func2.MULSon;
		sons.push_back(std::make_shared<MulFunction_Node>(func1));
		return MulFunction_Node(MULMUL, sons);   // 走构造函数重建，由构造函数自动化简
	}
	if(func2.Type==MULMUL){
		std::vector<std::shared_ptr<MulFunction_Node>> sons = func1.MULSon;
		sons.push_back(std::make_shared<MulFunction_Node>(func2));
		return MulFunction_Node(MULMUL, sons);   // 走构造函数重建，由构造函数自动化简
	}
	return MulFunction_Node(MULMUL, func1, func2);
}
MulFunction_Node operator/(const MulFunction_Node& func1, const MulFunction_Node& func2) {
	MulFunction_Node RECFUNC2 = REC(func2);
	return MulFunction_Node(MULMUL, func1, RECFUNC2);
}

MulFunction_Node& operator*=(MulFunction_Node& func1, const MulFunction_Node& func2) {
	if (func1.Type == MULMUL) {
		func1.MULSon.push_back(std::make_shared<MulFunction_Node>(func2));
		func1.Sort();
		return func1;
	}
	MulFunction_Node cope = func1;
	func1 = MulFunction_Node(MULMUL, cope, func2);
	return func1;
}
MulFunction_Node& operator/=(MulFunction_Node& func1,const MulFunction_Node& func2) {
	if (func1.Type == MULMUL) {
		func1.MULSon.push_back(std::make_shared<MulFunction_Node>(REC(func2)));
		func1.Sort();
		return func1;
	}
	func1 *= REC(func2);
	return func1;
}
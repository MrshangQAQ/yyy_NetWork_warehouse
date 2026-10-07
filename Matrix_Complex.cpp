#include"Matrix.h"
#include <cmath>
Matrix_Complex::Matrix_Complex(std::vector<Complex>data_, int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	int push_size = data_.size() < rowNum_*colNum_ ? data_.size() : rowNum_ * colNum_;
	for (int i = 0;i < push_size;i++) {
		data.push_back(data_[i]);
	}
	for (int i = push_size;i < rowNum_ * colNum_;i++) {
		data.push_back(Complex(0, 0));
	}
}
Matrix_Complex::Matrix_Complex(std::vector<float>data_, int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	int push_size = data_.size() < rowNum_ * colNum_ ? data_.size() : rowNum_ * colNum_;
	for (int i = 0;i < push_size;i++) {
		data.push_back(Complex(data_[i],0));
	}
	for (int i = push_size;i < rowNum_ * colNum_;i++) {
		data.push_back(Complex(0, 0));
	}
}
Matrix_Complex::Matrix_Complex(int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	for (int i = 0;i < rowNum_*colNum_;i++) {
		data.push_back(Complex(0, 0));
	}
}
Matrix_Complex::Matrix_Complex(int size) {
	if (size < 0) {
		throw std::runtime_error("negetive row or col");
	}
	if (size == 0) {
		return;
	}
	colNum_=rowNum_ = size;
	for (int i = 0;i < rowNum_*colNum_;i++) {
		if (i % size != i/size ) {
			data.push_back(Complex(0, 0));
		}
		else{
			data.push_back(Complex(1, 0));
		}
	}
}
Matrix_Complex::Matrix_Complex(MatrixView_Complex View) {
	for (int row = 0;row < View.RowLen;row++) {
		for (int col = 0;col < View.ColLen;col++) {
			data.push_back(*(View.first + (View.RowBegin + row) * View.ColNum_ + View.ColBegin + col));
		}
	}
	rowNum_ = View.RowLen;
	colNum_ = View.ColLen;
}
void Matrix_Complex::Swap_Row(int row1, int row2) {
	if (row1 == row2) {
		throw std::runtime_error("could not swap a row with itself");
	}
	if (row1 < 0 || row1 >= rowNum_ || row2 < 0 || row2 >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to swap the two rol");
	}
	InISwap_Row(row1, row2);
}
void Matrix_Complex::InISwap_Row(int row1, int row2) {
	for (int col = 0;col < colNum_;col++) {
		Swap(data[row1 * colNum_ + col], data[row2 * colNum_ + col]);
	}
}

void Matrix_Complex::Mutiple_Row(Complex times, int row) {
	if (row < 0 || row >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to mutiple the row");
	}
	InIMutiple_Row(times, row);
}
void Matrix_Complex::InIMutiple_Row(Complex times, int row) {
	for (int col = 0;col < colNum_;col++) {
		data[row * colNum_ + col] *= times;
	}
}

void Matrix_Complex::AddTimesTo_Row(int addRow, Complex times, int beAddedRow) {
	if (addRow == beAddedRow) {
		throw std::runtime_error("could not add a row to itself");
	}
	if (addRow < 0 || addRow >= rowNum_ || beAddedRow < 0 || beAddedRow >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to add the row to another");
	}
	InIAddTimesTo_Row(addRow, times, beAddedRow);
}
void Matrix_Complex::InIAddTimesTo_Row(int addRow, Complex times, int beAddedRow) {
	for (int col = 0;col < colNum_;col++) {
		data[beAddedRow * colNum_ + col] += times * data[addRow * colNum_ + col];
	}
}

void Matrix_Complex::Swap_Col(int col1, int col2) {
	if (col1 == col2) {
		throw std::runtime_error("could not swap a col with itself");
	}
	if (col1 < 0 || col1 >= colNum_ || col2 < 0 || col2 >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to swap the two col");
	}
	InISwap_Col(col1, col2);
}
void Matrix_Complex::InISwap_Col(int col1, int col2) {
	for (int row = 0;row < rowNum_;row++) {
		Swap(data[row * colNum_ + col1], data[row * colNum_ + col2]);
	}
}
void Matrix_Complex::Mutiple_Col(Complex times, int col) {
	if (col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to mutiple the col");
	}
	InIMutiple_Col(times, col);
}
void Matrix_Complex::InIMutiple_Col(Complex times, int col) {
	for (int row = 0;row < rowNum_;row++) {
		data[row * colNum_ + col] *= times;
	}
}
void Matrix_Complex::AddTimesTo_Col(int addCol, Complex times, int beAddedCol) {
	if (addCol == beAddedCol) {
		throw std::runtime_error("could not add a col to itself");
	}
	if (addCol < 0 || addCol >= colNum_ || beAddedCol < 0 || beAddedCol >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to add the col to another");
	}
	InIAddTimesTo_Col(addCol, times, beAddedCol);
}

void Matrix_Complex::InIAddTimesTo_Col(int addCol, Complex times, int beAddedCol) {
	for (int row = 0;row < rowNum_;row++) {
		data[row * colNum_ + beAddedCol] += times * data[row * colNum_ + addCol];
	}
}

Matrix_Complex GetTranslateMatrix(const Matrix_Complex& mat) {
	std::vector<Complex> newdata;
	for (int col = 0;col < mat.colNum_;col++) {
		for (int row = 0;row < mat.rowNum_;row++) {
			newdata.push_back(mat.data[row*mat.colNum_ + col].Conjugate());
		}
	}
	return Matrix_Complex(newdata, mat.colNum_, mat.rowNum_);
}
void Matrix_Complex::Translate() {
	if (rowNum_ == colNum_) {
		for (int row = 0;row < rowNum_;row++) {
			for (int col =1 ;col < row;col++) {
				Swap(data[row * colNum_ + col], data[col * rowNum_ + row]);
			}
		}
		return;
	}
	*this = std::move(GetTranslateMatrix(*this));
}

Complex Matrix_Complex::GetData(int row, int col) {
	if (row < 0 || row >= rowNum_ || col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to getdata");
	}
	return InIGetData(row, col);
}
void Matrix_Complex::SetData(int row, int col, Complex newdata) {
	if (row < 0 || row >= rowNum_ || col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to setdata");
	}
	InISetData(row, col, newdata);
}
Complex Matrix_Complex::InIGetData(int row,int col) {
	return data[row*colNum_+col];
}
void Matrix_Complex::InISetData(int row, int col, Complex newdata) {
	data[row * colNum_ + col] = newdata;
}
Matrix_Complex Matrix_Complex::InIGetMinor(int BeginRow, int EndRow, int BeginCol, int EndCol) {
	std::vector<Complex> newData;
	for (int row = BeginRow;row <= EndRow;row++) {
		for (int col = BeginCol;col <= EndCol;col++) {
			newData.push_back(data[row*colNum_+col]);
		}
	}
	return Matrix_Complex(newData, EndRow - BeginRow + 1, EndCol - BeginCol + 1);
}
Matrix_Complex Matrix_Complex::GetMinor(int BeginRow, int EndRow, int BeginCol, int EndCol) {
	if (BeginRow < 0 || EndRow >= rowNum_ || BeginCol < 0 || EndCol >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to get minor");
	}
	if (BeginRow > EndRow || BeginCol > EndCol) {
		throw std::runtime_error("begin should be less than end");
	}
	return InIGetMinor(BeginRow, EndRow, BeginCol, EndCol);
}

Matrix_Complex Matrix_Complex::InIGetMinorByVec(const std::vector<int>& RowVec, const std::vector<int>& ColVec) {
	std::vector<Complex>newData;
	for (int RowIndex = 0;RowIndex < RowVec.size();RowIndex++) {
		for (int ColIndex = 0; ColIndex < ColVec.size();ColIndex++) {
			newData.push_back(data[RowVec[RowIndex] * colNum_ + ColVec[ColIndex]]);
		}
	}
	return Matrix_Complex(newData, RowVec.size(), ColVec.size());
}
Matrix_Complex Matrix_Complex::GetMinorByVec(const std::vector<int>& RowVec, const std::vector<int>& ColVec) {
	std::vector<int> RowIn;//记录已经出现的Row
	std::vector<int> ColIn;//同上
	for (int i = 0;i < rowNum_;i++) {
		RowIn.push_back(0);
	}
	for (int i = 0;i < colNum_;i++) {
		ColIn.push_back(0);
	}
	for (int RowIndex = 0;RowIndex < RowVec.size();RowIndex++) {
		if (RowVec[RowIndex] < 0 || RowVec[RowIndex] >= rowNum_) {
			throw std::runtime_error("The row out-of-bounds access in try to get minor by vec");
		}
		if (RowIn[RowVec[RowIndex]] != 1) {
			RowIn[RowVec[RowIndex]] = 1;
		}
		else {
			throw std::runtime_error("Repetitive Row");
		}
	}
	for (int ColIndex = 0; ColIndex < ColVec.size();ColIndex++) {
		if (ColVec[ColIndex] < 0 || ColVec[ColIndex] >= colNum_) {
			throw std::runtime_error("The col out-of-bounds access in try to get minor by vec");
		}
		if (ColIn[ColVec[ColIndex]] != 1) {
			ColIn[ColVec[ColIndex]] = 1;
		}
		else {
			throw std::runtime_error("Repetitive Col");
		}
	}
	return InIGetMinorByVec(RowVec, ColVec);
}
void Matrix_Complex::NeglectSmall(double approximation) {
	Rational appro = approximation*approximation;
	for (int i = 0;i < rowNum_ * colNum_;i++) {
		if (data[i].ModulusSquare() < appro) {
			data[i] = Complex(0, 0);
		}
	}
}
int Matrix_Complex::Find_Main(int col) {
	if (col < 0 || col >= colNum_ ) {
		throw std::runtime_error("Out-of-bounds access in try to Find MainRow");
	}
	return InIFind_Main(col,0,0);
}
int Matrix_Complex::InIFind_Main(int col, int beginRow, double minModelSquare) {
	int searchRow = beginRow;
	Rational MaxModel(minModelSquare);
	Rational tmp(0.0);
	int MaxRow = -1;
	while (searchRow < rowNum_) {
		tmp = data[searchRow * colNum_ + col].ModulusSquare();
		if (tmp > MaxModel) {
			MaxModel = tmp;
			MaxRow = searchRow;
		}
		searchRow++;
	}
	return MaxRow;
}

MatrixOperationsReturn_Complex Matrix_Complex::GaussEliminat(double approximation) {
	if (approximation < 0) {
		throw std::runtime_error("negative approximation");
	}
	int nowCol = 0;//当前在消的列,其自然等于需要从第nowCol列开始消元,即nowRow
	int EliminatNums = rowNum_ < colNum_ ? rowNum_ : colNum_;//表示总共最多要消几次元，选行数列数中较小的那个
	int Swaptimes = 0;//记录交换发生的次数
	double approximationSquare = approximation * approximation;
	Rational approximationSquare_R(approximationSquare);
	//创建副本
	Matrix_Complex Duplicate(*this);
	//初始化记忆矩阵
	Matrix_Complex Memory(rowNum_, colNum_);
	for (int i = 0;i < EliminatNums;i++) {
		Memory.InISetData(i, i, Complex(1, 0));
	}

	for (nowCol;nowCol < EliminatNums;nowCol++) {
		int MainRow = Duplicate.InIFind_Main(nowCol, nowCol, approximationSquare);

		if (MainRow == -1) {
			for (int col = nowCol;col < colNum_;col++) {
				Duplicate.InISwap_Col(nowCol, col);
				Memory.InISwap_Col(nowCol, col);
				Swaptimes++;
				MainRow = Duplicate.InIFind_Main(nowCol, nowCol, approximationSquare);
				if (MainRow != -1) {
					break;
				}
				if (col == colNum_ - 1) {
					return MatrixOperationsReturn_Complex{ std::move(Duplicate),std::move(Memory),Swaptimes };
				}
			}
		}

		if (MainRow != nowCol) {
			Duplicate.InISwap_Row(MainRow, nowCol);
			Memory.InISwap_Row(MainRow, nowCol);
			Swaptimes++;
		}

		for (int row = nowCol + 1;row < rowNum_;row++) {
			if (Duplicate.InIGetData(row, nowCol).ModulusSquare() > approximationSquare_R) {
				Complex times = -Duplicate.InIGetData(row, nowCol) / Duplicate.InIGetData(nowCol, nowCol);
				Duplicate.InIAddTimesTo_Row(nowCol, times, row);
				Memory.InIAddTimesTo_Row(nowCol, times, row);
			}
		}
	}
	return MatrixOperationsReturn_Complex{ std::move(Duplicate),std::move(Memory),Swaptimes };
}

MatrixOperationsReturn_Complex Matrix_Complex::Diagonalization(double approximation) {
	MatrixOperationsReturn_Complex DiaData = GaussEliminat(approximation);
	int nowcol= rowNum_ < colNum_ ? rowNum_ : colNum_;//从该列开始，依次往上消元
	Rational approximationSquare(approximation * approximation);
	Complex tmp_Times(0, 0);
	for (int col = nowcol - 1;col >= 0;col--) {
		if (DiaData.ResultMat.GetData(col, col).ModulusSquare() < approximationSquare) {
			continue;
		}
		for (int row = col-1;row >= 0;row--) {
			if (DiaData.ResultMat.InIGetData(row, col).ModulusSquare() < approximationSquare) {
				continue;
			}
			tmp_Times =-DiaData.ResultMat.InIGetData(row,col)/DiaData.ResultMat.InIGetData(col, col);
			DiaData.ResultMat.InIAddTimesTo_Row(col, tmp_Times, row);
			DiaData.Memory.InIAddTimesTo_Row(col, tmp_Times, row);
		}
	}
	return DiaData;
}

Complex Matrix_Complex::GetDeterminant(float approximation) {
	if (rowNum_ != colNum_) {
		throw std::runtime_error("just could get determinate from phalanx");
	}
	MatrixOperationsReturn_Complex DetData = Diagonalization(approximation);
	Complex result(1, 0);
	for (int i = 0;i < rowNum_;i++) {
		result *= DetData.ResultMat.InIGetData(i, i);
	}
	if (DetData.Swaptimes % 2 == 1) {
		return -result;
	}
	return result;
}
GetInverseData_Complex Matrix_Complex::GetRank(float approximation) {
	GetInverseData_Complex RankData = { Diagonalization(approximation),0 };
	float approximationSquare = approximation * approximation;
	int maxRank = rowNum_ < colNum_ ? rowNum_ : colNum_;
	for (int i = 0;i < maxRank;i++) {
		if (RankData.OperaData.ResultMat.InIGetData(i, i).ModulusSquare() >= approximationSquare) {
			RankData.rank++;
		}
	}
	return RankData;
}
OrthogonalDecomposition_Result Matrix_Complex::OrthogonalDecomposition() {
	if (rowNum_ != colNum_) {
		throw std::runtime_error("This matrix isn't symmetric matrix");
	}
	Matrix_Complex translate(GetTranslateMatrix(*this));
	for (int i = 0;i < rowNum_ * colNum_;i++) {
		if (translate.data[i] != data[i]||data[i].imag!=Rational(0,1)) {
			throw std::runtime_error("This matrix isn't symmetric matrix");
		}
	}
	return InIOrthogonalDecomposition();
}
double Matrix_Complex::GetModle() {
	if (rowNum_ != 1 && colNum_ != 1) {
		throw std::runtime_error("only vector(matrix with one row or one col) could get Modle");
	}
	return InIGetModle();
}
double Matrix_Complex::InIGetModle() {
	Rational sum(0, 1);
	for (int i = 0;i < rowNum_*colNum_;i++) {
		sum += data[i].ModulusSquare();
	}
	return sqrt((double)sum);
}

OrthogonalDecomposition_Result Matrix_Complex::InIOrthogonalDecomposition() {
	int n = rowNum_;
	// 第一步：把有理数矩阵一次性转成 double 工作区（特征值天生是无理数，中间计算必须浮点）
	std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			A[i][j] = (double)data[i * colNum_ + j].real;
		}
	}
	// Q 从单位矩阵开始，一路累积所有正交变换，最后每列就是一个特征向量
	std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0.0));
	for (int i = 0;i < n;i++) {
		Q[i][i] = 1.0;
	}

	// 第二步：Householder 三对角化（有限步精确完成）
	// 每一列用一次"镜像反射"，把主对角线以下元素一次性清零
	for (int k = 0;k < n - 2;k++) {
		// 取第 k 列主对角线以下部分，算模长
		double normx = 0.0;
		for (int i = k + 1;i < n;i++) {
			normx += A[i][k] * A[i][k];
		}
		normx = std::sqrt(normx);
		if (normx < 1e-300) {
			continue;//这一列已全是零，无需反射
		}
		// α 与 A[k+1][k] 反号，避免相减丢失有效数字
		double alpha = (A[k + 1][k] > 0.0) ? -normx : normx;
		// 反射向量 v = x - α·e1（仅 k+1..n-1 分量非零）
		std::vector<double> v(n, 0.0);
		v[k + 1] = A[k + 1][k] - alpha;
		for (int i = k + 2;i < n;i++) {
			v[i] = A[i][k];
		}
		double vnorm2 = 0.0;
		for (int i = k + 1;i < n;i++) {
			vnorm2 += v[i] * v[i];
		}
		double beta = 2.0 / vnorm2;//H = I - β·v·vᵀ
		// 秩一更新：H·A·H = A - v·pᵀ - p·vᵀ，其中 p = β·A·v - (β/2)·(vᵀ·(βA·v))·v
		std::vector<double> p(n, 0.0);
		for (int i = k;i < n;i++) {
			double sum = 0.0;
			for (int j = k + 1;j < n;j++) {
				sum += A[i][j] * v[j];
			}
			p[i] = beta * sum;
		}
		double vtp = 0.0;
		for (int i = k + 1;i < n;i++) {
			vtp += v[i] * p[i];
		}
		double corr = 0.5 * beta * vtp;
		for (int i = k;i < n;i++) {
			p[i] -= corr * v[i];
		}
		// A ← A - v·pᵀ - p·vᵀ（对称，只更新右下角子块并对称写回）
		for (int i = k;i < n;i++) {
			for (int j = i;j < n;j++) {
				double sub = v[i] * p[j] + p[i] * v[j];
				A[i][j] -= sub;
				if (i != j) {
					A[j][i] -= sub;
				}
			}
		}
		// 数值保险：显式把第 k 列对角线以下清零为 (α, 0, ..., 0)
		A[k + 1][k] = alpha;
		A[k][k + 1] = alpha;
		for (int i = k + 2;i < n;i++) {
			A[i][k] = 0.0;
			A[k][i] = 0.0;
		}
		// 累积 Q ← Q·H = Q - β·(Q·v)·vᵀ
		std::vector<double> t(n, 0.0);
		for (int i = 0;i < n;i++) {
			double sum = 0.0;
			for (int j = k + 1;j < n;j++) {
				sum += Q[i][j] * v[j];
			}
			t[i] = beta * sum;
		}
		for (int i = 0;i < n;i++) {
			for (int j = k + 1;j < n;j++) {
				Q[i][j] -= t[i] * v[j];
			}
		}
	}

	// 三对角矩阵只需主对角元 d 和次对角元 e 两个数组
	std::vector<double> d(n), e(n - 1, 0.0);
	for (int i = 0;i < n;i++) {
		d[i] = A[i][i];
	}
	for (int i = 0;i < n - 1;i++) {
		e[i] = A[i + 1][i];
	}

	// 第三步：带 Wilkinson 位移的隐式 QR 迭代（凸起追逐）
	const double EPS = 1e-12;//收敛阈值
	int maxIter = 200 * n * n + 50;//迭代上限，防止死循环
	int iter = 0;
	int m = n - 1;//当前待处理块的右下角
	while (m > 0 && iter < maxIter) {
		// e[m-1] 已足够小：该次对角元收敛，块收缩
		if (std::fabs(e[m - 1]) <= EPS * (std::fabs(d[m - 1]) + std::fabs(d[m]))) {
			e[m - 1] = 0.0;
			m--;
			continue;
		}
		// 找未收敛块的上界 l（e[l-1] 已收敛，或 l 就是左上角）
		int l = m - 1;
		while (l > 0 && std::fabs(e[l - 1]) > EPS * (std::fabs(d[l - 1]) + std::fabs(d[l]))) {
			l--;
		}
		// Wilkinson 位移：右下 2×2 子块 [[d[m-1],e[m-1]],[e[m-1],d[m]]] 中靠近 d[m] 的特征值
		double delta = (d[m - 1] - d[m]) / 2.0;
		double mu = d[m] - e[m - 1] * e[m - 1] / (delta + (delta >= 0.0 ? 1.0 : -1.0) * std::sqrt(delta * delta + e[m - 1] * e[m - 1]));

		// 对块 [l, m] 做一步隐式 QR：一串 Givens 旋转把"凸起"从左推到右
		double x = d[l] - mu;
		double z = e[l];
		for (int k = l;k < m;k++) {
			double r = std::hypot(x, z);
			double c = x / r;
			double s = z / r;
			// 对称双旋转 2×2 块 [[d[k],e[k]],[e[k],d[k+1]]] 的闭式更新公式
			double a = d[k], b = e[k], dd = d[k + 1];
			d[k] = c * c * a + 2.0 * c * s * b + s * s * dd;
			d[k + 1] = s * s * a - 2.0 * c * s * b + c * c * dd;
			e[k] = c * s * (dd - a) + (c * c - s * s) * b;
			// 上一步留下的凸起在此被吸收：e[k-1] 变为 r
			if (k > l) {
				e[k - 1] = r;
			}
			// 累积 Q ← Q·G：只更新第 k、k+1 两列
			for (int i = 0;i < n;i++) {
				double qk = Q[i][k];
				double qk1 = Q[i][k + 1];
				Q[i][k] = c * qk + s * qk1;
				Q[i][k + 1] = -s * qk + c * qk1;
			}
			// 把凸起向右推进一格
			if (k < m - 1) {
				x = e[k];
				z = s * e[k + 1];
				e[k + 1] = c * e[k + 1];
			}
		}
		iter++;
	}
	if (m > 0) {
		throw std::runtime_error("OrthogonalDecomposition: QR iteration did not converge");
	}

	// 第四步：特征值按从大到小排序，Q 的列同步交换
	for (int i = 0;i < n - 1;i++) {
		int best = i;
		for (int j = i + 1;j < n;j++) {
			if (d[j] > d[best]) {
				best = j;
			}
		}
		if (best != i) {
			double tmpd = d[i];
			d[i] = d[best];
			d[best] = tmpd;
			for (int r2 = 0;r2 < n;r2++) {
				double tmpq = Q[r2][i];
				Q[r2][i] = Q[r2][best];
				Q[r2][best] = tmpq;
			}
		}
	}

	// 第五步：double 转回 Rational 并装进结果（特征值是无理数，转回必有舍入，这是可接受的）
	OrthogonalDecomposition_Result result;
	result.Spectrum.resize(n);
	for (int i = 0;i < n;i++) {
		result.Spectrum[i] = Complex(d[i], 0.0);
	}
	std::vector<Complex> qdata(n * n);
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			qdata[i * colNum_ + j] = Complex(Q[i][j], 0.0);
		}
	}
	result.Q = Matrix_Complex(qdata, n, n);
	return result;
}

Matrix_Complex GetInverseMatrix(const Matrix_Complex& mat, float approximation) {
	if (mat.rowNum_ != mat.colNum_) {
		throw std::runtime_error("just could get InverseMatrix from phalanx");
	}
	Matrix_Complex work = mat;//拷贝副本，因为 GetRank 是非 const 成员函数，无法在 const 对象上调用
	GetInverseData_Complex InvData = work.GetRank(approximation);
	if (InvData.rank < mat.rowNum_) {
		throw std::runtime_error("could get inverseMatrix because the matrix isn't full of rank");
	}
	for (int i = 0;i < mat.rowNum_;i++) {
		InvData.OperaData.Memory.InIMutiple_Row(Complex(1, 0) / InvData.OperaData.ResultMat.InIGetData(i, i), i);
	}
	return InvData.OperaData.Memory;
}

Matrix_Lambda GetLambdaMatrix(const Matrix_Complex& mat) {
	if (mat.rowNum_ != mat.colNum_) {
		throw std::runtime_error("just could get LambdaMatrix from phalanx");
	}
	std::vector<Polynomial> LamData;
	LamData.resize(mat.rowNum_ * mat.colNum_, Polynomial(Complex()));
	for (int i = 0;i < mat.rowNum_ * mat.colNum_;i++) {
		if (i%mat.rowNum_ == i/mat.rowNum_) {
			LamData[i] = Polynomial({mat.data[i],Complex(-1)});
		}
		else {
			LamData[i] = Polynomial(mat.data[i]);
		}
	}
	return Matrix_Lambda(LamData, mat.rowNum_, mat.colNum_);
}






Matrix_Complex operator+(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		std::runtime_error("mismatched matrix");
	}
	std::vector<Complex> newdata;
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		newdata.push_back(matrix1.data[i] + matrix2.data[i]);
	}
	return Matrix_Complex(newdata, matrix1.rowNum_, matrix1.colNum_);
}
Matrix_Complex& operator+=(Matrix_Complex& matrix1, const Matrix_Complex& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		std::runtime_error("mismatched matrix");
	}
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		matrix1.data[i] += matrix2.data[i];
	}
	return matrix1;
}
Matrix_Complex operator-(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		std::runtime_error("mismatched matrix");
	}
	std::vector<Complex> newdata;
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		newdata.push_back(matrix1.data[i] - matrix2.data[i]);
	}
	return Matrix_Complex(newdata, matrix1.rowNum_, matrix1.colNum_);
}
Matrix_Complex& operator-=(Matrix_Complex& matrix1, const Matrix_Complex& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		std::runtime_error("mismatched matrix");
	}
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		matrix1.data[i] -= matrix2.data[i];
	}
	return matrix1;
}
Matrix_Complex operator*(const Matrix_Complex& matrix1, const Matrix_Complex& matrix2) {
	if (matrix1.colNum_ != matrix2.rowNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int rows = matrix1.rowNum_, inner = matrix1.colNum_, cols = matrix2.colNum_;
	// 一次分配到位，避免 push_back 反复扩容
	std::vector<Complex> newdata(rows * cols, Complex(0, 0));
	// ikj 顺序：i(行) → k(内维) → j(列)，B 与 C 都按行连续访问，缓存友好
	for (int i = 0; i < rows; i++) {
		int iABase = i * inner;   // 强度削减：行基址提前算好，避免内层重复乘法
		int iCBase = i * cols;
		for (int k = 0; k < inner; k++) {
			const Complex& a = matrix1.data[iABase + k];
			if (a == Complex(0, 0)) continue; // 跳过零元素（容差比较，若在意严格精确可去掉此行）
			int kBBase = k * cols;
			for (int j = 0; j < cols; j++) {
				newdata[iCBase + j] += a * matrix2.data[kBBase + j];
			}
		}
	}
	return Matrix_Complex(newdata, rows, cols);
}
Matrix_Complex operator*(Complex com, const Matrix_Complex& mat) {
	std::vector<Complex> newdata = mat.data;
	int size = mat.rowNum_ * mat.colNum_;
	for (int i = 0;i < size;i++) {
		newdata[i] *= com;
	}
	return Matrix_Complex(newdata, mat.rowNum_, mat.colNum_);
}
Matrix_Complex operator*(const Matrix_Complex& mat, Complex com) {
	return com * mat;
}
Matrix_Complex& operator*=(Matrix_Complex& mat, Complex com) {

	int size = mat.rowNum_ * mat.colNum_;
	for (int i = 0;i < size;i++) {
		mat.data[i] *= com;
	}
	return mat;
}
std::ostream& operator<<(std::ostream& os, const Matrix_Complex& mat) {
	for (int row = 0;row < mat.rowNum_;row++) {
		for (int col = 0;col < mat.colNum_;col++) {
			os <<std::left<<std::setw(15) << mat.data[row * mat.colNum_ + col];
		}
		os << '\n';
	}
	return os;
}
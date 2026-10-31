#include"Matrix.h"
#include <cmath>	// std::fabs / std::sqrt / std::hypot
#include <utility>	// std::swap / std::move

Matrix_Double::Matrix_Double(std::vector<double>data_, int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	int totalNum = rowNum_ * colNum_;
	data = std::move(data_);
	data.resize(totalNum, 0.0);
}
Matrix_Double::Matrix_Double(std::vector<float>data_, int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	int totalNum = rowNum_ * colNum_;
	int push_size = (int)data_.size() < totalNum ? (int)data_.size() : totalNum;
	data.resize(totalNum, 0.0);
	for (int i = 0;i < push_size;i++) {
		data[i] = data_[i];
	}
}
Matrix_Double::Matrix_Double(int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	data.resize(colNum_ * rowNum_, 0);
}
Matrix_Double::Matrix_Double(int size) {
	if (size < 0) {
		throw std::runtime_error("negetive row or col");
	}
	if (size == 0) {
		rowNum_ = 0;
		colNum_ = 0;
		return;//空矩阵：显式初始化行列数后返回
	}
	colNum_ = rowNum_ = size;
	data.resize(colNum_ * rowNum_, 0);
	for (int i = 0;i < rowNum_ * colNum_;i++) {
		//行号 = i/size，列号 = i%size，两者相等即位于对角线上
		if (i % size == i / size) {
			data[i]=1.0;
		}
	}
}

void Matrix_Double::Swap_Row(int row1, int row2) {
	if (row1 == row2) {
		throw std::runtime_error("could not swap a row with itself");
	}
	if (row1 < 0 || row1 >= rowNum_ || row2 < 0 || row2 >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to swap the two rows");
	}
	InISwap_Row(row1, row2);
}
void Matrix_Double::InISwap_Row(int row1, int row2) {
	for (int col = 0;col < colNum_;col++) {
		std::swap(data[row1 * colNum_ + col], data[row2 * colNum_ + col]);
	}
}

void Matrix_Double::Mutiple_Row(double times, int row) {
	if (row < 0 || row >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to mutiple the row");
	}
	InIMutiple_Row(times, row);
}
void Matrix_Double::InIMutiple_Row(double times, int row) {
	for (int col = 0;col < colNum_;col++) {
		data[row * colNum_ + col] *= times;
	}
}

void Matrix_Double::AddTimesTo_Row(int addRow, double times, int beAddedRow) {
	if (addRow == beAddedRow) {
		throw std::runtime_error("could not add a row to itself");
	}
	if (addRow < 0 || addRow >= rowNum_ || beAddedRow < 0 || beAddedRow >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to add the row to another");
	}
	InIAddTimesTo_Row(addRow, times, beAddedRow);
}
void Matrix_Double::InIAddTimesTo_Row(int addRow, double times, int beAddedRow) {
	for (int col = 0;col < colNum_;col++) {
		//倍加变换：beAddedRow 行 += times × addRow 行
		data[beAddedRow * colNum_ + col] += times * data[addRow * colNum_ + col];
	}
}

void Matrix_Double::Swap_Col(int col1, int col2) {
	if (col1 == col2) {
		throw std::runtime_error("could not swap a col with itself");
	}
	if (col1 < 0 || col1 >= colNum_ || col2 < 0 || col2 >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to swap the two cols");
	}
	InISwap_Col(col1, col2);
}
void Matrix_Double::InISwap_Col(int col1, int col2) {
	for (int row = 0;row < rowNum_;row++) {
		std::swap(data[row * colNum_ + col1], data[row * colNum_ + col2]);
	}
}
void Matrix_Double::Mutiple_Col(double times, int col) {
	if (col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to mutiple the col");
	}
	InIMutiple_Col(times, col);
}
void Matrix_Double::InIMutiple_Col(double times, int col) {
	for (int row = 0;row < rowNum_;row++) {
		data[row * colNum_ + col] *= times;
	}
}
void Matrix_Double::AddTimesTo_Col(int addCol, double times, int beAddedCol) {
	if (addCol == beAddedCol) {
		throw std::runtime_error("could not add a col to itself");
	}
	if (addCol < 0 || addCol >= colNum_ || beAddedCol < 0 || beAddedCol >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to add the col to another");
	}
	InIAddTimesTo_Col(addCol, times, beAddedCol);
}
void Matrix_Double::InIAddTimesTo_Col(int addCol, double times, int beAddedCol) {
	for (int row = 0;row < rowNum_;row++) {
		//倍加变换：beAddedCol 列 += times × addCol 列
		data[row * colNum_ + beAddedCol] += times * data[row * colNum_ + addCol];
	}
}

Matrix_Double GetTranslateMatrix(const Matrix_Double& mat) {
	//按列优先依次取出每个元素，恰好构成转置矩阵的行主序数据
	std::vector<double> newdata(mat.rowNum_*mat.colNum_);
	for (int col = 0;col < mat.colNum_;col++) {
		double* New_Base_Address = &newdata[col*mat.rowNum_];
		const double* Mat_Base_Address = &mat.data[col];
		for (int row = 0;row < mat.rowNum_;row++) {
			New_Base_Address[row]=Mat_Base_Address[row*mat.colNum_];
		}
	}
	return Matrix_Double(std::move(newdata), mat.colNum_, mat.rowNum_);//行列数互换
}
void Matrix_Double::Translate() {
	if (rowNum_ == colNum_) {
		//方阵：只遍历上三角（col > row），每个元素与镜像位置交换一次即可
		for (int row = 0;row < rowNum_;row++) {
			for (int col = row + 1;col < colNum_;col++) {
				std::swap(data[row * colNum_ + col], data[col * colNum_ + row]);
			}
		}
		return;
	}
	//非方阵：行列数会互换，无法原地完成，重建矩阵后移动赋值
	*this = std::move(GetTranslateMatrix(*this));
}

double Matrix_Double::GetData(int row, int col) const {
	if (row < 0 || row >= rowNum_ || col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to getdata");
	}
	return InIGetData(row, col);
}
void Matrix_Double::SetData(int row, int col, double newdata) {
	if (row < 0 || row >= rowNum_ || col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to setdata");
	}
	InISetData(row, col, newdata);
}
double Matrix_Double::InIGetData(int row, int col) const {
	return data[row * colNum_ + col];
}
void Matrix_Double::InISetData(int row, int col, double newdata) {
	data[row * colNum_ + col] = newdata;
}

void Matrix_Double::NeglectSmall(double approximation) {
	for (int i = 0;i < rowNum_ * colNum_;i++) {
		if (std::fabs(data[i]) < approximation) {
			data[i] = 0.0;
		}
	}
}

int Matrix_Double::Find_Main(int col) const {
	if (col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to Find MainRow");
	}
	return InIFind_Main(col, 0, 0.0);
}
int Matrix_Double::InIFind_Main(int col, int beginRow, double minModel) const {
	int searchRow = beginRow;
	double MaxModel = minModel;//当前找到的最大绝对值，初值为阈值
	int MaxRow = -1;//没找到时保持 -1
	while (searchRow < rowNum_) {
		double tmp = std::fabs(data[searchRow * colNum_ + col]);
		if (tmp > MaxModel) {
			MaxModel = tmp;
			MaxRow = searchRow;
		}
		searchRow++;
	}
	return MaxRow;
}

MatrixOperationsReturn_Double Matrix_Double::GaussEliminat(double approximation) {
	if (approximation < 0) {
		throw std::runtime_error("negative approximation");
	}
	int EliminatNums = rowNum_ < colNum_ ? rowNum_ : colNum_;//总共最多消元 min(行数,列数) 次
	int Swaptimes = 0;//记录交换发生的次数（用于计算行列式）
	//创建副本，原矩阵保持不变；Memory 记录对副本做过的所有行列变换
	Matrix_Double Duplicate(*this);
	Matrix_Double Memory(rowNum_, colNum_);
	for (int i = 0;i < EliminatNums;i++) {
		Memory.InISetData(i, i, 1.0);
	}

	for (int nowCol = 0;nowCol < EliminatNums;nowCol++) {
		int MainRow = Duplicate.InIFind_Main(nowCol, nowCol, approximation);

		if (MainRow == -1) {
			//当前列从 nowCol 行往下找不到主元，尝试与右侧的列交换
			for (int col = nowCol + 1;col < colNum_;col++) {
				Duplicate.InISwap_Col(nowCol, col);
				Memory.InISwap_Col(nowCol, col);
				Swaptimes++;
				MainRow = Duplicate.InIFind_Main(nowCol, nowCol, approximation);
				if (MainRow != -1) {
					break;
				}
			}
			if (MainRow == -1) {
				//所有列都无法提供主元，说明矩阵秩不足，提前结束消元
				return MatrixOperationsReturn_Double{ std::move(Duplicate),std::move(Memory),Swaptimes };
			}
		}

		if (MainRow != nowCol) {
			//把主元所在行交换到当前行
			Duplicate.InISwap_Row(MainRow, nowCol);
			Memory.InISwap_Row(MainRow, nowCol);
			Swaptimes++;
		}

		for (int row = nowCol + 1;row < rowNum_;row++) {
			if (std::fabs(Duplicate.InIGetData(row, nowCol)) > approximation) {
				//用主元行把该行这一列消成 0
				double times = -Duplicate.InIGetData(row, nowCol) / Duplicate.InIGetData(nowCol, nowCol);
				Duplicate.InIAddTimesTo_Row(nowCol, times, row);
				Memory.InIAddTimesTo_Row(nowCol, times, row);
			}
		}
	}
	return MatrixOperationsReturn_Double{ std::move(Duplicate),std::move(Memory),Swaptimes };
}

MatrixOperationsReturn_Double Matrix_Double::Diagonalization(double approximation) {
	MatrixOperationsReturn_Double DiaData = GaussEliminat(approximation);
	int nowCol = rowNum_ < colNum_ ? rowNum_ : colNum_;//从该列开始，依次往上把上三角也消成 0
	for (int col = nowCol - 1;col >= 0;col--) {
		if (std::fabs(DiaData.ResultMat.InIGetData(col, col)) < approximation) {
			continue;//主元太小，跳过该列
		}
		for (int row = col - 1;row >= 0;row--) {
			if (std::fabs(DiaData.ResultMat.InIGetData(row, col)) < approximation) {
				continue;
			}
			double times = -DiaData.ResultMat.InIGetData(row, col) / DiaData.ResultMat.InIGetData(col, col);
			DiaData.ResultMat.InIAddTimesTo_Row(col, times, row);
			DiaData.Memory.InIAddTimesTo_Row(col, times, row);
		}
	}
	return DiaData;
}

OrthogonalDecomposition_Result_Double Matrix_Double::OrthogonalDecomposition() {
	if (rowNum_ != colNum_) {
		throw std::runtime_error("This matrix isn't symmetric matrix");
	}
	//转置后逐元素比较：完全相等才算对称
	Matrix_Double translate(GetTranslateMatrix(*this));
	for (int i = 0;i < rowNum_ * colNum_;i++) {
		if (translate.data[i] != data[i]) {
			throw std::runtime_error("This matrix isn't symmetric matrix");
		}
	}
	return InIOrthogonalDecomposition();
}

OrthogonalDecomposition_Result_Double Matrix_Double::InIOrthogonalDecomposition() {
	int n = rowNum_;
	if (n == 0) {
		//空矩阵：直接返回空结果，避免后面构造长度 -1 的数组
		OrthogonalDecomposition_Result_Double emptyResult;
		emptyResult.Q = Matrix_Double(0, 0);
		return emptyResult;
	}
	// 把矩阵内容拷入 double 工作区 A（本类数据本身就是 double，直接复制即可）
	std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			A[i][j] = data[i * colNum_ + j];
		}
	}
	// Q 从单位矩阵开始，一路累积所有正交变换，最后每列就是一个特征向量
	std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0.0));
	for (int i = 0;i < n;i++) {
		Q[i][i] = 1.0;
	}

	// 第一步：Householder 三对角化（有限步精确完成）
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

	// 第二步：带 Wilkinson 位移的隐式 QR 迭代（凸起追逐）
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

	// 第三步：特征值按从大到小排序，Q 的列同步交换
	for (int i = 0;i < n - 1;i++) {
		int best = i;
		for (int j = i + 1;j < n;j++) {
			if (d[j] > d[best]) {
				best = j;
			}
		}
		if (best != i) {
			std::swap(d[i], d[best]);
			for (int r2 = 0;r2 < n;r2++) {
				std::swap(Q[r2][i], Q[r2][best]);
			}
		}
	}

	// 第四步：把结果装进返回结构体（Q 的各列与 Spectrum 一一对应）
	OrthogonalDecomposition_Result_Double result;
	result.Spectrum.resize(n);
	for (int i = 0;i < n;i++) {
		result.Spectrum[i] = d[i];
	}
	std::vector<double> qdata(n * n);
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			qdata[i * n + j] = Q[i][j];
		}
	}
	result.Q = Matrix_Double(qdata, n, n);
	return result;
}
Matrix_Double Matrix_Double::InIGetMinor(int BeginRow, int EndRow, int BeginCol, int EndCol) const {
	int NewRowNum = EndRow - BeginRow + 1;
	int NewColNum = EndCol - BeginCol + 1;
	std::vector<double> newData(NewRowNum*NewColNum);
	for (int row = 0;row < NewRowNum;row++) {
		double* New_Base_Address = &newData[row * NewColNum];
		const double* Mat_Base_Address = &data[(row+BeginRow)*colNum_+BeginCol];
		for (int col = 0;col < NewColNum;col++) {
			New_Base_Address[col]= Mat_Base_Address[col];
		}
	}
	return Matrix_Double(std::move(newData), NewRowNum, NewColNum);
}

Matrix_Double Matrix_Double::GetMinor(int BeginRow, int EndRow, int BeginCol, int EndCol) const {
	if (BeginRow < 0 || EndRow >= rowNum_ || BeginCol < 0 || EndCol >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to get minor");
	}
	if (BeginRow > EndRow || BeginCol > EndCol) {
		throw std::runtime_error("begin should be less than end");
	}
	return InIGetMinor(BeginRow, EndRow, BeginCol, EndCol);
}

Matrix_Double Matrix_Double::InIGetMinorByVec(const std::vector<int>& RowVec, const std::vector<int>& ColVec) const {
	std::vector<double>newData;
	for (int RowIndex = 0;RowIndex < RowVec.size();RowIndex++) {
		for (int ColIndex = 0; ColIndex < ColVec.size();ColIndex++) {
			newData.push_back(data[RowVec[RowIndex] * colNum_ + ColVec[ColIndex]]);
		}
	}
	return Matrix_Double(newData, RowVec.size(), ColVec.size());
}

Matrix_Double Matrix_Double::GetMinorByVec(const std::vector<int>& RowVec, const std::vector<int>& ColVec) const {
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
Matrix_Double Matrix_Double::GetMinorByVec(const std::vector<int>& ColVec) const {
	std::vector<double>newdata(ColVec.size() * rowNum_,0.0);
	for (int row = 0;row < rowNum_;row++) {
		int i_row_index = row * colNum_;
		int i_new_index = row * ColVec.size();
		for (int i = 0;i < ColVec.size();i++) {
			newdata[i_new_index + i] = data[i_row_index + ColVec[i]];
		}
	}
	return Matrix_Double(std::move(newdata), rowNum_, ColVec.size());
}
void Matrix_Double::Random(double ave, double std) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> normal(ave, std);
	int size = colNum_ * rowNum_;
	for (int i = 0;i < size;i++) {
		data[i] = normal(gen);
	}
}
void Matrix_Double::Rotato_180() {
	int reverseTimes = data.size() / 2;
	int end = data.size() - 1;
	for (int i = 0;i < reverseTimes;i++) {
		std::swap(data[i], data[end - i]);
	}
}






Matrix_Double GetInverseMatrix(const Matrix_Double& mat, double approximation) {
	if (mat.rowNum_ != mat.colNum_) {
		throw std::runtime_error("just could get InverseMatrix from phalanx");
	}
	Matrix_Double work = mat;//拷贝副本，因为 Diagonalization 是非 const 成员函数，无法在 const 对象上调用
	MatrixOperationsReturn_Double InvData = work.Diagonalization(approximation);
	//若任何对角元接近 0，说明矩阵不满秩，不可逆
	for (int i = 0;i < mat.rowNum_;i++) {
		if (std::fabs(InvData.ResultMat.InIGetData(i, i)) < approximation) {
			throw std::runtime_error("could not get inverseMatrix because the matrix isn't full of rank");
		}
	}
	//消元过程中 Memory 与副本同步变换，满足 Memory·A = 对角形 D；
	//把 Memory 的每一行除以对应对角元（左乘 D⁻¹），即得 A⁻¹
	for (int i = 0;i < mat.rowNum_;i++) {
		InvData.Memory.InIMutiple_Row(1.0 / InvData.ResultMat.InIGetData(i, i), i);
	}
	return InvData.Memory;
}

Matrix_Double GetDoubleMatrix(const Matrix_Complex& mat) {
	std::vector<double> newdata;
	int totalNum = mat.rowNum_ * mat.colNum_;
	newdata.reserve(totalNum);
	for (int i = 0;i < totalNum;i++) {
		newdata.push_back((double)mat.data[i].real);//只取实部，虚部舍弃
	}
	return Matrix_Double(std::move(newdata), mat.rowNum_, mat.colNum_);
}
Matrix_Complex GetComplexMatrix(const Matrix_Double& mat) {
	std::vector<Complex> newdata;
	int totalNum = mat.rowNum_ * mat.colNum_;
	newdata.reserve(totalNum);
	for (int i = 0;i < totalNum;i++) {
		newdata.push_back(Complex(mat.data[i], 0.0));
	}
	return Matrix_Complex(std::move(newdata), mat.rowNum_, mat.colNum_);
}
Matrix_Double Hadamard(const Matrix_Double& mat1,const Matrix_Double& mat2) {
	if (mat1.colNum_ != mat2.colNum_ || mat1.rowNum_ != mat2.rowNum_) {
		throw std::runtime_error("mismatched hadamard");
	}
	std::vector<double> newdata;
	int size = mat1.colNum_ * mat1.rowNum_;
	newdata.resize(size,0);
	for (int i = 0;i < size;i++) {
		newdata[i] = mat1.data[i] * mat2.data[i];
	}
	return Matrix_Double(std::move(newdata), mat1.rowNum_, mat1.colNum_);
}

std::vector<Matrix_Double> Hadamard(const std::vector<Matrix_Double>& mat1, const std::vector<Matrix_Double>& mat2) {
	if (mat1.size() != mat2.size()) {
		throw std::runtime_error("mismatched hadamard");
	}
	std::vector<Matrix_Double> result(mat1.size());
	for (int i = 0;i < result.size();i++) {
		result[i] = Hadamard(mat1[i], mat2[i]);
	}
	return result;
}
// ==================== 逐元素运算与激活函数（供神经网络使用） ====================

/// <summary>逐元素施加一元函数：返回与 mat 同形的新矩阵，其中 result(i,j) = f(mat(i,j))。</summary>
/// <param name="mat">输入矩阵（只读，不修改）</param>
/// <param name="f">一元标量函数指针，接受并返回 double</param>
/// <returns>与 mat 同形的新矩阵</returns>
/// <exception cref="std::runtime_error">f 为空指针时抛出</exception>
Matrix_Double MapMatrix(const Matrix_Double& mat, double(*f)(double)) {
	if (f == nullptr) {
		throw std::runtime_error("null function pointer");
	}
	int size = mat.rowNum_ * mat.colNum_;
	std::vector<double> newdata(size, 0.0);
	for (int i = 0; i < size; i++) {
		newdata[i] = f(mat.data[i]);	// 行主序下标一一对应，逐元素施加
	}
	return Matrix_Double(std::move(newdata), mat.rowNum_, mat.colNum_);
}


Matrix_Double RowSums(const Matrix_Double& mat) {
	int rows = mat.rowNum_, cols = mat.colNum_;
	std::vector<double> newdata(rows, 0.0);
	for (int i = 0; i < rows; i++) {
		double sum = 0.0;
		int base = i * cols;			// 强度削减：行基址提到内层循环外
		for (int j = 0; j < cols; j++) {
			sum += mat.data[base + j];
		}
		newdata[i] = sum;
	}
	return Matrix_Double(std::move(newdata), rows, 1);
}

/// <summary>给每一列加上同一个偏置列向量（广播）：result(i,j) = mat(i,j) + bias(i,0)。</summary>
/// <param name="mat">输入矩阵 n×m</param>
/// <param name="bias">偏置列向量 n×1</param>
/// <returns>n×m 矩阵</returns>
/// <exception cref="std::runtime_error">bias 不是列向量、或行数与 mat 不符时抛出</exception>
Matrix_Double AddBias(const Matrix_Double& mat, const Matrix_Double& bias) {
	if (bias.colNum_ != 1 || bias.rowNum_ != mat.rowNum_) {
		throw std::runtime_error("mismatched addbias");
	}
	int rows = mat.rowNum_, cols = mat.colNum_;
	std::vector<double> newdata(rows * cols, 0.0);
	for (int i = 0; i < rows; i++) {
		double b = bias.data[i];		// 偏置只跟行号有关，提到列循环外
		int base = i * cols;
		for (int j = 0; j < cols; j++) {
			newdata[base + j] = mat.data[base + j] + b;
		}
	}
	return Matrix_Double(std::move(newdata), rows, cols);
}

Matrix_Double Softmax(const Matrix_Double& z) {
	int rows = z.GetRowNum(), cols = z.GetColNum();
	std::vector<double> out(rows * cols, 0.0);
	for (int j = 0; j < cols; j++) {                 // 逐列（逐样本）
		double maxv = z.data[j];                     // 数值稳定：先减去该列最大值，防 exp 溢出
		for (int i = 1; i < rows; i++)
			if (z.data[i * cols + j] > maxv) maxv = z.data[i * cols + j];
		double sum = 0.0;
		for (int i = 0; i < rows; i++) {
			double e = std::exp(z.data[i * cols + j] - maxv);
			out[i * cols + j] = e;
			sum += e;
		}
		for (int i = 0; i < rows; i++) out[i * cols + j] /= sum;
	}
	return Matrix_Double(std::move(out), rows, cols);
}
/// <summary>对矩阵四周补零（padding>0）或裁剪（padding&lt;0）。padding=0 时返回原矩阵副本。
/// 用于卷积的边界处理；负 padding 语义与 PyTorch F.pad 一致。</summary>
/// <param name="mat">输入矩阵</param>
/// <param name="padding">四周补零/裁剪的圈数：正数补零，负数裁剪，0 不变</param>
/// <returns>尺寸为 (row+2·padding) × (col+2·padding) 的新矩阵</returns>
Matrix_Double GetPaddingMatrix(const Matrix_Double& mat, int padding) {
	if (padding == 0) {
		return mat;
	}
	const int newRowNums = mat.rowNum_ + 2 * padding;
	const int newColNums = mat.colNum_ + 2 * padding;
	if (newRowNums <= 0 || newColNums <= 0) {
		throw std::runtime_error("padding too large: empty result");
	}
	std::vector<double> newdata(newRowNums * newColNums, 0.0);
	for (int row = 0; row < newRowNums; row++) {
		const int srcRow = row - padding;   // 正 padding：四周留出边框；负 padding：跳过被裁的行
		if (srcRow < 0 || srcRow >= mat.rowNum_) continue;   // 落进补零区
		double* dst = &newdata[row * newColNums];
		const double* src = &mat.data[srcRow * mat.colNum_];
		for (int col = 0; col < newColNums; col++) {
			const int srcCol = col - padding;
			if (srcCol < 0 || srcCol >= mat.colNum_) continue;
			dst[col] = src[srcCol];
		}
	}
	return Matrix_Double(std::move(newdata), newRowNums, newColNums);
}

// 尺寸应为 ((n-1)*stride+1)：元素落到下标 0,stride,2*stride,...,(n-1)*stride，
// 旧写法 n*stride 会多出 stride-1 行/列零尾巴，使后续卷积输出比预期小 (stride-1)
Matrix_Double GetDilateMatrix(const Matrix_Double& mat, int stride) {
	if (stride == 1) {
		return mat;
	}
	if (stride <= 0) {
		throw std::runtime_error("negative or zero stride");
	}
	const int newRowNums = (mat.rowNum_ - 1) * stride + 1;
	const int newColNums = (mat.colNum_ - 1) * stride + 1;
	std::vector newdata(newColNums * newRowNums, 0.0);
	for (int row = 0;row < mat.rowNum_;row ++) {
		int i_row_index_new = row * stride * newColNums;
		int i_row_index_mat = row * mat.colNum_;
		for (int col = 0;col < mat.colNum_;col ++) {
			newdata[i_row_index_new + col * stride] = mat.data[i_row_index_mat + col];
		}
	}
	return Matrix_Double(std::move(newdata), newRowNums, newColNums);
}


//池化这个矩阵，这是类内调用，方便返回值，如果不想改变原矩阵，需要手动拷贝
AntiPoll_Data Matrix_Double::MaxPoll(int pollsize) {
	if (pollsize <= 0) {
		throw std::runtime_error("pollsize must be positive");
	}
	const int outRows = rowNum_ / pollsize;   // 整数除法向下取整：不足一个窗口的行被丢弃
	const int outCols = colNum_ / pollsize;
	if (outRows <= 0 || outCols <= 0) {
		throw std::runtime_error("matrix is smaller than one polling window");
	}
	if (pollsize == 1) {
		return { std::vector<int>(rowNum_ * colNum_,0.0),rowNum_,colNum_ };
	}
	std::vector<double> newdata(outRows * outCols, 0.0);
	std::vector<int> index_list(outRows * outCols, 0);
	for (int r = 0; r < outRows; r++) {
		const int baseRow = r * pollsize;
		for (int c = 0; c < outCols; c++) {
			const int baseCol = c * pollsize;
			double bestVal = data[baseRow * colNum_ + baseCol];   // 以窗口左上角为初始冠军
			int bestOffset = 0;
			for (int dr = 0; dr < pollsize; dr++) {
				const double* rowPtr = &data[(baseRow + dr) * colNum_ + baseCol];   // 行首地址提到内层循环外，省下标乘法
				for (int dc = 0; dc < pollsize; dc++) {
					if (rowPtr[dc] > bestVal) {              // 严格 >：并列时取先扫到的
						bestVal = rowPtr[dc];
						bestOffset = dr * pollsize + dc;
					}
				}
			}
			newdata[r * outCols + c] = bestVal;
			index_list[r * outCols + c] = bestOffset;
		}
	}
	AntiPoll_Data result = {index_list,rowNum_,colNum_};
	rowNum_ = outRows;
	colNum_ = outCols;
	data = std::move(newdata);
	return result;
}

Matrix_Double GetAntiMaxPoll(const Matrix_Double& pollmat, int pollsize, const AntiPoll_Data& antipoll_data) {
	if (pollsize <= 0) {
		throw std::runtime_error("pollsize must be positive");
	}
	const std::vector<int>& index_list = antipoll_data.index_list;
	const int outRows = pollmat.GetRowNum();
	const int outCols = pollmat.GetColNum();
	if ((int)index_list.size() != outRows * outCols) {
		throw std::runtime_error("index_list size mismatched with the polling matrix");
	}
	if (pollsize == 1) {
		return pollmat;
	}
	// 默认全 0：非冠军位置梯度天然为 0（前向未使用过它们）
	std::vector<double> newdata((size_t)antipoll_data.ordRowNums*antipoll_data.ordColNums, 0.0);

	const std::vector<double>& src = pollmat.GetInternalData();
	const int windowSize = pollsize * pollsize;
	for (int r = 0; r < outRows; r++) {
		double* New_Base_Address = &newdata[(size_t)r*pollsize*antipoll_data.ordColNums];
		const double* Src_Base_Address = &src[(size_t)r * outCols];
		const int* Offset_Base_Address = &index_list[(size_t)r * outCols];
		for (int c = 0; c < outCols; c++) {
			if (Offset_Base_Address[c] < 0 || Offset_Base_Address[c] >= windowSize) {
				throw std::runtime_error("index out of window range");
			}
			const int dr = Offset_Base_Address[c] / pollsize;   // 窗口内行
			const int dc = Offset_Base_Address[c] % pollsize;   // 窗口内列
			New_Base_Address[dr*antipoll_data.ordColNums + (c*pollsize+dc)] = Src_Base_Address[c];
		}
	}
	// 构造签名为 (data, row, col)：行数=ordRowNums、列数=ordColNums
	return Matrix_Double(std::move(newdata), antipoll_data.ordRowNums, antipoll_data.ordColNums);
}

Matrix_Double Convolution(const Matrix_Double& mat,const Matrix_Double& kernel, int padding, int stride) {
	if (stride <= 0) {
		throw std::runtime_error("stride=0 or negative stride");
	}
	const int outRows = (mat.rowNum_ - kernel.rowNum_ + 2 * padding) / stride + 1;
	const int outCols = (mat.colNum_ - kernel.colNum_ + 2 * padding) / stride + 1;
	if (kernel.rowNum_ > mat.rowNum_+2*padding || kernel.colNum_ > mat.colNum_+2*padding
		|| kernel.rowNum_ <= 0 || kernel.colNum_ <= 0) {
		throw std::runtime_error("mismatched convolution");
	}
	std::vector<double> newdata(outRows*outCols, 0.0);

	if (padding == 0) {
		for (int row = 0;row < outRows;row++) {
			for (int col = 0;col < outCols;col++) {
				int i_index_mat = (row * mat.colNum_ + col)*stride;   // 窗口左上角基址
				double sum = 0.0;
				for (int exrow = 0; exrow < kernel.rowNum_; exrow++) {
					const double* matRow = &mat.data[i_index_mat + exrow * mat.colNum_];
					const double* kerRow = &kernel.data[exrow * kernel.colNum_];
					for (int excol = 0; excol < kernel.colNum_; excol++) {
						sum += matRow[excol] * kerRow[excol];
					}
				}
				newdata[row * outCols + col] = sum;
			}
		}
	}
	else {
		Matrix_Double newmat = GetPaddingMatrix(mat, padding);
		for (int row = 0;row < outRows;row++) {
			for (int col = 0;col < outCols;col++) {
				int i_index_mat = (row * newmat.colNum_ + col)*stride;   // 窗口左上角基址
				double sum = 0.0;
				for (int exrow = 0; exrow < kernel.rowNum_; exrow++) {
					const double* matRow = &newmat.data[i_index_mat + exrow * newmat.colNum_];
					const double* kerRow = &kernel.data[exrow * kernel.colNum_];
					for (int excol = 0; excol < kernel.colNum_; excol++) {
						sum += matRow[excol] * kerRow[excol];
					}
				}
				newdata[row*outCols+col] = sum;
			}
		}
	}
	return Matrix_Double(std::move(newdata),outRows,outCols);
}

std::vector<Matrix_Double> Convolution(const std::vector<Matrix_Double>& mat, const Matrix_Double& kernel, int padding, int stride) {
	std::vector<Matrix_Double> copy(mat.size());
	for (int i = 0;i < mat.size();i++) {
		copy[i] = Convolution(mat[i], kernel, padding, stride);
	}
	return copy;
}
//将三阶，四阶张量与矩阵相互转化
Matrix_Double GetFlattenMatrixFromBatch(const std::vector<Matrix_Double>& batch) {
	if (batch.size() == 0) {
		return Matrix_Double(0);
	}
	int RowNums = batch[0].data.size();
	std::vector<double> newdata(RowNums * batch.size());
	for (int col = 0;col < batch.size();col++) {
		if (batch[col].colNum_!=batch[0].colNum_||batch[col].rowNum_!=batch[0].rowNum_) {
			throw std::runtime_error("Non-standard batch,the vector's matrix isn't all aligned");
		}
		double* row_Base_ptr = &newdata[col];
		const double* batch_Base_ptr = &batch[col].data[0];
		for (int row = 0;row < RowNums;row++) {
			row_Base_ptr[row*batch.size()] = batch_Base_ptr[row];//沿列读取
		}
	}
	return Matrix_Double(std::move(newdata), RowNums, batch.size());
}
std::vector<Matrix_Double> UnFlattenToBatch(const Matrix_Double& mat, int h, int w) {
	const int D = h * w;
	std::vector<Matrix_Double> out(mat.colNum_);
	for (int s = 0; s < mat.colNum_; s++) {
		std::vector<double> d(D);
		for (int i = 0; i < D; i++) {
		d[i] = mat.GetInternalData()[i * mat.colNum_ + s];
		}
		out[s] = Matrix_Double(std::move(d), h, w);
	}
	return out;
}
Matrix_Double Flatten(const Matrix_Double& mat) {
	return Matrix_Double(mat.GetInternalData(),
		mat.GetRowNum() * mat.GetColNum(), 1);
}
/// <summary>Flatten 的逆：把 (H·W)×1 向量还原成 h×w 特征图。</summary>
Matrix_Double Unflatten(const Matrix_Double& vec, int h, int w) {
	return Matrix_Double(vec.GetInternalData(), h, w);
}
Matrix_Double FlattenChannels(const std::vector<std::vector<Matrix_Double>>& batch) {
	const int N = (int)batch.size();
	const int C = (int)batch[0].size();
	const int H = batch[0][0].GetRowNum();
	const int W = batch[0][0].GetColNum();
	const int D = C * H * W;                 // 拍平后的特征维
	std::vector<double> out((size_t)D * N, 0.0);
	for (int s = 0; s < N; s++)
		for (int c = 0; c < C; c++) {
			const std::vector<double>& src = batch[s][c].GetInternalData();
			for (int hw = 0; hw < H * W; hw++) {
				int row = c * H * W + hw;                       // 通道优先，排成特征行的下标
				out[(size_t)row * N + s] = src[hw];             // 第 s 列 = 第 s 个样本
			}
		}
	return Matrix_Double(std::move(out), D, N);
}
std::vector<std::vector<Matrix_Double>> UnflattenChannels(const Matrix_Double& grad, int C, int H, int W) {
	const int N = grad.GetColNum();
	std::vector<std::vector<Matrix_Double>> out(
		N, std::vector<Matrix_Double>(C, Matrix_Double(H, W)));
	for (int s = 0; s < N; s++)
		for (int c = 0; c < C; c++) {
			std::vector<double> d((size_t)H * W);
			for (int hw = 0; hw < H * W; hw++) {
				int row = c * H * W + hw;                       // 和 FlattenChannels 完全对称
				d[hw] = grad.GetInternalData()[(size_t)row * N + s];
			}
			out[s][c] = Matrix_Double(std::move(d), H, W);
		}
	return out;
}




Matrix_Double operator+(const Matrix_Double& matrix1, const Matrix_Double& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int size = matrix1.rowNum_ * matrix2.colNum_;
	std::vector<double> newdata(size);
	for (int i = 0;i < size;i++) {
		newdata[i]=matrix1.data[i] + matrix2.data[i];
	}
	return Matrix_Double(std::move(newdata), matrix1.rowNum_, matrix1.colNum_);
}
Matrix_Double& operator+=(Matrix_Double& matrix1, const Matrix_Double& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		matrix1.data[i] += matrix2.data[i];
	}
	return matrix1;
}
Matrix_Double operator-(const Matrix_Double& matrix1, const Matrix_Double& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int size = matrix1.rowNum_ * matrix2.colNum_;
	std::vector<double> newdata(size);
	for (int i = 0;i < size;i++) {
		newdata[i] = matrix1.data[i] - matrix2.data[i];
	}
	return Matrix_Double(std::move(newdata), matrix1.rowNum_, matrix1.colNum_);
}
Matrix_Double& operator-=(Matrix_Double& matrix1, const Matrix_Double& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		matrix1.data[i] -= matrix2.data[i];
	}
	return matrix1;
}
Matrix_Double operator*(const Matrix_Double& matrix1, const Matrix_Double& matrix2) {
	if (matrix1.colNum_ != matrix2.rowNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int rows = matrix1.rowNum_, inner = matrix1.colNum_, cols = matrix2.colNum_;
	//一次分配到位，避免 push_back 反复扩容
	std::vector<double> newdata(rows * cols);
	//外部分块：每次圈住 64 列的 B 面板（约 256 KB），使其驻留 L2，
	//这样内层重复遍历同一块 B 时打的是二级缓存而不是主存。
	int JB = 64;
	for (int jj = 0; jj < cols; jj += JB) {//相当于一次性读取256kb的数据
		const int jend = (jj + JB < cols) ? (jj + JB) : cols;//末尾行，防止越界
		const int jb = jend - jj;//遍历的部分
		double* iCBase0 = &newdata[jj];
		const double* kBBase0 = &matrix2.data[jj];
		// 主体：一次算 4 行 × 8 列 = 32 个输出，32 个累加器全程驻留寄存器，k 循环内不读写内存。
		// 相比原来的 1×8：同一条 B 的 cache line（64 字节 = 8 个 double）原来只喂 8 次乘加，
		// 现在被 4 行共用、喂 32 次，算术强度由 4.5 B/FLOP 降到 1.5 B/FLOP，内存通路不再是瓶颈。
		const int iMain = rows & ~3;// 向下取整到 4 的倍数；余下 0~3 行交给后面的 1×8 尾部
#pragma omp parallel for if(rows > 64)
		for (int i = 0; i < iMain; i += 4) {
			// 4 行的 A 行首地址（行主序：第 r 行从 matrix1.data[(i+r)*inner] 开始）
			const double* __restrict aRow0 = &matrix1.data[(size_t)(i + 0) * inner];
			const double* __restrict aRow1 = &matrix1.data[(size_t)(i + 1) * inner];
			const double* __restrict aRow2 = &matrix1.data[(size_t)(i + 2) * inner];
			const double* __restrict aRow3 = &matrix1.data[(size_t)(i + 3) * inner];
			// 4 行的 C 行首地址
			double* __restrict cRow0 = iCBase0 + (size_t)(i + 0) * cols;
			double* __restrict cRow1 = iCBase0 + (size_t)(i + 1) * cols;
			double* __restrict cRow2 = iCBase0 + (size_t)(i + 2) * cols;
			double* __restrict cRow3 = iCBase0 + (size_t)(i + 3) * cols;

			int j = 0;
			for (; j + 8 <= jb; j += 8) {
				// 累加器命名 r{行}c{列}，共 32 个，整个 k 循环内只住寄存器
				double r0c0 = 0, r0c1 = 0, r0c2 = 0, r0c3 = 0, r0c4 = 0, r0c5 = 0, r0c6 = 0, r0c7 = 0;
				double r1c0 = 0, r1c1 = 0, r1c2 = 0, r1c3 = 0, r1c4 = 0, r1c5 = 0, r1c6 = 0, r1c7 = 0;
				double r2c0 = 0, r2c1 = 0, r2c2 = 0, r2c3 = 0, r2c4 = 0, r2c5 = 0, r2c6 = 0, r2c7 = 0;
				double r3c0 = 0, r3c1 = 0, r3c2 = 0, r3c3 = 0, r3c4 = 0, r3c5 = 0, r3c6 = 0, r3c7 = 0;
				// 用指针自增代替每次 k*cols 的乘法（强度削减），同时避免 int 溢出
				const double* __restrict bPtr = kBBase0 + j;
				for (int k = 0; k < inner; k++) {
					// B 的 8 个元素取进寄存器后供 4 行复用，这是把内存流量摊薄的关键
					const double b0 = bPtr[0], b1 = bPtr[1], b2 = bPtr[2], b3 = bPtr[3];
					const double b4 = bPtr[4], b5 = bPtr[5], b6 = bPtr[6], b7 = bPtr[7];
					const double x0 = aRow0[k], x1 = aRow1[k], x2 = aRow2[k], x3 = aRow3[k];
					r0c0 += x0 * b0; r0c1 += x0 * b1; r0c2 += x0 * b2; r0c3 += x0 * b3;
					r0c4 += x0 * b4; r0c5 += x0 * b5; r0c6 += x0 * b6; r0c7 += x0 * b7;
					r1c0 += x1 * b0; r1c1 += x1 * b1; r1c2 += x1 * b2; r1c3 += x1 * b3;
					r1c4 += x1 * b4; r1c5 += x1 * b5; r1c6 += x1 * b6; r1c7 += x1 * b7;
					r2c0 += x2 * b0; r2c1 += x2 * b1; r2c2 += x2 * b2; r2c3 += x2 * b3;
					r2c4 += x2 * b4; r2c5 += x2 * b5; r2c6 += x2 * b6; r2c7 += x2 * b7;
					r3c0 += x3 * b0; r3c1 += x3 * b1; r3c2 += x3 * b2; r3c3 += x3 * b3;
					r3c4 += x3 * b4; r3c5 += x3 * b5; r3c6 += x3 * b6; r3c7 += x3 * b7;
					bPtr += cols;// 步长保持 = cols，且 k 累加顺序不变（CPU/GPU 逐位对拍要求）
				}
				// 整个 k 循环结束后一次性写回 4 行 × 8 列，消除内层里的“读 C + 写 C”
				cRow0[j + 0] = r0c0; cRow0[j + 1] = r0c1; cRow0[j + 2] = r0c2; cRow0[j + 3] = r0c3;
				cRow0[j + 4] = r0c4; cRow0[j + 5] = r0c5; cRow0[j + 6] = r0c6; cRow0[j + 7] = r0c7;
				cRow1[j + 0] = r1c0; cRow1[j + 1] = r1c1; cRow1[j + 2] = r1c2; cRow1[j + 3] = r1c3;
				cRow1[j + 4] = r1c4; cRow1[j + 5] = r1c5; cRow1[j + 6] = r1c6; cRow1[j + 7] = r1c7;
				cRow2[j + 0] = r2c0; cRow2[j + 1] = r2c1; cRow2[j + 2] = r2c2; cRow2[j + 3] = r2c3;
				cRow2[j + 4] = r2c4; cRow2[j + 5] = r2c5; cRow2[j + 6] = r2c6; cRow2[j + 7] = r2c7;
				cRow3[j + 0] = r3c0; cRow3[j + 1] = r3c1; cRow3[j + 2] = r3c2; cRow3[j + 3] = r3c3;
				cRow3[j + 4] = r3c4; cRow3[j + 5] = r3c5; cRow3[j + 6] = r3c6; cRow3[j + 7] = r3c7;
			}
			// j 尾巴：仅当 jb 不是 8 的倍数时进入（最多 7 列），4 行一起算
			for (; j < jb; j++) {
				const double* __restrict bPtr = kBBase0 + j;
				double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
				for (int k = 0; k < inner; k++) {
					const double b = *bPtr;
					s0 += aRow0[k] * b;
					s1 += aRow1[k] * b;
					s2 += aRow2[k] * b;
					s3 += aRow3[k] * b;
					bPtr += cols;
				}
				cRow0[j] = s0; cRow1[j] = s1; cRow2[j] = s2; cRow3[j] = s3;
			}
		}
		// 尾部：rows 不是 4 的倍数时，剩余 0~3 行沿用原来的 1×8 微内核，保证不漏行
		for (int i = iMain; i < rows; i++) {
			const double* __restrict iABase = &matrix1.data[(size_t)i * inner];
			double* __restrict iCBase = iCBase0 + (size_t)i * cols;

			//j 方向一次算 8 个输出：8 个累加器全程住在寄存器里，整个 k 循环不再读写内存。
			//之所以取 8，是因为 8 个 double = 64 字节 = 一条 cache line，每次从 B 取到的
			//数据能被完整利用；若只算 4 个则每次只吃半条 line，另一半要等到下一轮 j 块才用，
			//那时早已被挤出缓存，等于白白多读一倍流量。
			int j = 0;
			for (; j + 8 <= jb; j += 8) {
				double c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
				//用指针自增代替每次 k*cols 的乘法（强度削减），同时避免 int 溢出
				const double* __restrict bPtr = kBBase0 + j;
				for (int k = 0; k < inner; k++) {
					const double a = iABase[k];
					c0 += a * bPtr[0];
					c1 += a * bPtr[1];
					c2 += a * bPtr[2];
					c3 += a * bPtr[3];
					c4 += a * bPtr[4];
					c5 += a * bPtr[5];
					c6 += a * bPtr[6];
					c7 += a * bPtr[7];
					bPtr += cols;
				}
				//整个 k 循环结束后只写一次，消除旧实现内层里的“读 C + 写 C”
				iCBase[j + 0] = c0; iCBase[j + 1] = c1;
				iCBase[j + 2] = c2; iCBase[j + 3] = c3;
				iCBase[j + 4] = c4; iCBase[j + 5] = c5;
				iCBase[j + 6] = c6; iCBase[j + 7] = c7;
			}
			//尾巴：仅当 jb 不是 8 的倍数时进入（jb 最大 64，故最多处理 7 列）
			for (; j < jb; j++) {
				const double* __restrict bPtr = kBBase0 + j;
				double s = 0;
				for (int k = 0; k < inner; k++) {
					s += iABase[k] * (*bPtr);
					bPtr += cols;
				}
				iCBase[j] = s;
			}
		}
	}
	return Matrix_Double(std::move(newdata), rows, cols);
}
Matrix_Double operator*(double db, const Matrix_Double& mat) {
	std::vector<double> newdata = mat.data;
	int size = mat.rowNum_ * mat.colNum_;
	for (int i = 0;i < size;i++) {
		newdata[i] *= db;
	}
	return Matrix_Double(std::move(newdata), mat.rowNum_, mat.colNum_);
}
Matrix_Double operator*(const Matrix_Double& mat, double db) {
	return db * mat;
}
Matrix_Double& operator*=(Matrix_Double& mat, double db) {

	int size = mat.rowNum_ * mat.colNum_;
	for (int i = 0;i < size;i++) {
		mat.data[i] *= db;
	}
	return mat;
}
std::ostream& operator<<(std::ostream& os, const Matrix_Double& mat) {
	for (int row = 0;row < mat.rowNum_;row++) {
		for (int col = 0;col < mat.colNum_;col++) {
			os << std::left << std::setw(15) << mat.data[row * mat.colNum_ + col];
		}
		os << '\n';
	}
	return os;
}

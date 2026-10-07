#include"Matrix.h"

Matrix_Lambda::Matrix_Lambda(std::vector<Polynomial>data_, int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	int push_size = data_.size() < rowNum_ * colNum_ ? data_.size() : rowNum_ * colNum_;
	for (int i = 0;i < push_size;i++) {
		data.push_back(data_[i]);
	}
	for (int i = push_size;i < rowNum_ * colNum_;i++) {
		data.push_back(Polynomial(Complex(0, 0)));//不足的部分补零多项式
	}
}
Matrix_Lambda::Matrix_Lambda(int row, int col) {
	if (row < 0 || col < 0) {
		throw std::runtime_error("negetive row or col");
	}
	rowNum_ = row;
	colNum_ = col;
	for (int i = 0;i < rowNum_ * colNum_;i++) {
		data.push_back(Polynomial(Complex(0, 0)));
	}
}
Matrix_Lambda::Matrix_Lambda(int size) {
	if (size < 0) {
		throw std::runtime_error("negetive row or col");
	}
	if (size == 0) {
		return;
	}
	colNum_ = rowNum_ = size;
	for (int i = 0;i < rowNum_ * colNum_;i++) {
		// i % size 是列号，i / size 是行号，相等即位于对角线上
		if (i % size != i / size) {
			data.push_back(Polynomial(Complex(0, 0)));
		}
		else {
			data.push_back(Polynomial(Complex(1, 0)));//对角线填常数多项式 1
		}
	}
}

void Matrix_Lambda::Swap_Row(int row1, int row2) {
	if (row1 == row2) {
		throw std::runtime_error("could not swap a row with itself");
	}
	if (row1 < 0 || row1 >= rowNum_ || row2 < 0 || row2 >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to swap the two rol");
	}
	InISwap_Row(row1, row2);
}
void Matrix_Lambda::InISwap_Row(int row1, int row2) {
	for (int col = 0;col < colNum_;col++) {
		Swap(data[row1 * colNum_ + col], data[row2 * colNum_ + col]);
	}
}

void Matrix_Lambda::Mutiple_Row(const Polynomial& times, int row) {
	if (row < 0 || row >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to mutiple the row");
	}
	InIMutiple_Row(times, row);
}
void Matrix_Lambda::InIMutiple_Row(const Polynomial& times, int row) {
	for (int col = 0;col < colNum_;col++) {
		data[row * colNum_ + col] *= times;
	}
}

void Matrix_Lambda::AddTimesTo_Row(int addRow, const Polynomial& times, int beAddedRow) {
	if (addRow == beAddedRow) {
		throw std::runtime_error("could not add a row to itself");
	}
	if (addRow < 0 || addRow >= rowNum_ || beAddedRow < 0 || beAddedRow >= rowNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to add the row to another");
	}
	InIAddTimesTo_Row(addRow, times, beAddedRow);
}
void Matrix_Lambda::InIAddTimesTo_Row(int addRow, const Polynomial& times, int beAddedRow) {
	for (int col = 0;col < colNum_;col++) {
		// 语义：beAddedRow += times × addRow
		data[beAddedRow * colNum_ + col] += times * data[addRow * colNum_ + col];
	}
}

void Matrix_Lambda::Swap_Col(int col1, int col2) {
	if (col1 == col2) {
		throw std::runtime_error("could not swap a col with itself");
	}
	if (col1 < 0 || col1 >= colNum_ || col2 < 0 || col2 >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to swap the two col");
	}
	InISwap_Col(col1, col2);
}
void Matrix_Lambda::InISwap_Col(int col1, int col2) {
	for (int row = 0;row < rowNum_;row++) {
		Swap(data[row * colNum_ + col1], data[row * colNum_ + col2]);
	}
}

void Matrix_Lambda::Mutiple_Col(const Polynomial& times, int col) {
	if (col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to mutiple the col");
	}
	InIMutiple_Col(times, col);
}
void Matrix_Lambda::InIMutiple_Col(const Polynomial& times, int col) {
	for (int row = 0;row < rowNum_;row++) {
		data[row * colNum_ + col] *= times;
	}
}

void Matrix_Lambda::AddTimesTo_Col(int addCol, const Polynomial& times, int beAddedCol) {
	if (addCol == beAddedCol) {
		throw std::runtime_error("could not add a col to itself");
	}
	if (addCol < 0 || addCol >= colNum_ || beAddedCol < 0 || beAddedCol >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in trying to add the col to another");
	}
	InIAddTimesTo_Col(addCol, times, beAddedCol);
}
void Matrix_Lambda::InIAddTimesTo_Col(int addCol, const Polynomial& times, int beAddedCol) {
	for (int row = 0;row < rowNum_;row++) {
		// 语义：beAddedCol += times × addCol
		data[row * colNum_ + beAddedCol] += times * data[row * colNum_ + addCol];
	}
}
int Matrix_Lambda::Find_Main(int col) {
	if (col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to Find MainRow");
	}
	return InIFind_Main(col, 0);
}
int Matrix_Lambda::InIFind_Main(int col, int beginRow) {
	int minRow = -1;
	for (int row = beginRow;row < rowNum_;row++) {
		if (data[row * colNum_ + col].IsEmpty()) {
			continue;
		}
		if (minRow == -1 || data[row * colNum_ + col].Coefficient.size() < data[row * colNum_ + col].Coefficient.size()) {
			minRow = row;
		}
	}
	return minRow;
}
Matrix_Lambda Matrix_Lambda::GetTranslateMatrix() {
	std::vector<Polynomial>newdata;

	for (int col = 0;col < colNum_;col++) {
		for (int row = 0;row < rowNum_;row++) {
			newdata.push_back(data[row * colNum_ + col]);
		}
	}
	return Matrix_Lambda(newdata, colNum_, rowNum_);
}
void Matrix_Lambda::Translate() {
	if (rowNum_ == colNum_) {
		for (int row = 1;row < rowNum_;row++) {
			for (int col = 0;col < row;col++) {
				Swap(data[row * colNum_ + col], data[col * rowNum_ + row]);
			}
		}
		return;
	}
	*this = std::move(GetTranslateMatrix());
}

Polynomial Matrix_Lambda::GetData(int row, int col) {
	if (row < 0 || row >= rowNum_ || col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to getdata");
	}
	return InIGetData(row, col);
}
void Matrix_Lambda::SetData(int row, int col, Polynomial newdata) {
	if (row < 0 || row >= rowNum_ || col < 0 || col >= colNum_) {
		throw std::runtime_error("Out-of-bounds access in try to setdata");
	}
	InISetData(row, col, newdata);
}
Polynomial Matrix_Lambda::InIGetData(int row, int col) {
	return data[row * colNum_ + col];
}
void Matrix_Lambda::InISetData(int row, int col, Polynomial newdata) {
	data[row * colNum_ + col] = newdata;
}

void Matrix_Lambda::Diagonalization() {
	if (rowNum_ != colNum_) {
		throw std::runtime_error("the lambda matirx couldn't diagonlization");
	}
	int SwapTimes = 0;//记录交换的次数
	for (int times = 0;times < 2;times++) {
		//总共做两次，直接做一次，转置后再做一次
		for (int col = 0;col < colNum_;col++) {
			int Main_Row = InIFind_Main(col, col);
			if (Main_Row == -1) {
				for (int nextCol = col + 1;nextCol < colNum_;nextCol++) {
					InISwap_Col(col, nextCol);
					SwapTimes++;
					Main_Row = InIFind_Main(col, col);
					if (Main_Row != -1) {
						break;
					}
				}
				if (Main_Row == -1) {
					goto ProcessEnd;
				}
			}
			//求主元所在行
			if (Main_Row != col) {
				InISwap_Row(col, Main_Row);
				SwapTimes++;
			}
			//开始对row=col+1到row=rowNum_-1消元
			for (int row = col + 1;row < rowNum_;row++) {
				PolyDivison_Result Div = data[row * colNum_ + col] / data[col * colNum_ + col];
				InIAddTimesTo_Row(col, -1 * Div.Quotient, row);
				while (!data[row * colNum_ + col].IsEmpty()) {
					InISwap_Row(col, row);
					SwapTimes++;
					PolyDivison_Result Div = data[row * colNum_ + col] / data[col * colNum_ + col];
					Div.Quotient *= -1;
					InIAddTimesTo_Row(col, Div.Quotient, row);
				}
			}
		}
		//如果是第一次循环，将之转置,继续进行第二次循环
		ProcessEnd:
		if (times == 0) {
			Translate();
		}
	}
	for (int i = 0;i < rowNum_-1;i++) {
		if (!IsExactlyDivisible(data[i * colNum_ + i], data[(i + 1) * colNum_ + i+1])) {
			PolyGDC_Result gdc = GetGCD(data[i * colNum_ + i], data[(i + 1) * colNum_ + i + 1]);
			Polynomial NewI = gdc.GDC;
			Polynomial NewIPuls1 = data[(i + 1) * colNum_ + i + 1] * ((data[i * colNum_ + i] / gdc.GDC).Quotient);
			Swap(NewI, data[i * colNum_ + i]);
			Swap(NewIPuls1, data[(i + 1) * colNum_ + i + 1]);
		}
	}//将之化为smith标准型
}
Matrix_Lambda GetDiagonalization(const Matrix_Lambda& mat) {
	Matrix_Lambda result = mat;//拷贝副本，避免修改原矩阵
	result.Diagonalization();//原地对角化副本
	return result;
}
Polynomial Matrix_Lambda::GetDeterminant() {
	Matrix_Lambda DiaMat = GetDiagonalization(*this);
	Polynomial pol(Complex(1));
	for (int i = 0;i < rowNum_;i++) {
		pol *= DiaMat.data[i * colNum_ + i];
	}
	return pol;
}
Complex Matrix_Lambda::GetPolynomailValue(Complex value) {
	return GetDeterminant().CalculatedValue(value);
}


Matrix_Lambda operator+(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	std::vector<Polynomial> newdata;
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		newdata.push_back(matrix1.data[i] + matrix2.data[i]);
	}
	return Matrix_Lambda(newdata, matrix1.rowNum_, matrix1.colNum_);
}
Matrix_Lambda& operator+=(Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		matrix1.data[i] += matrix2.data[i];
	}
	return matrix1;
}
Matrix_Lambda operator-(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	std::vector<Polynomial> newdata;
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		newdata.push_back(matrix1.data[i] - matrix2.data[i]);
	}
	return Matrix_Lambda(newdata, matrix1.rowNum_, matrix1.colNum_);
}
Matrix_Lambda& operator-=(Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2) {
	if (matrix1.rowNum_ != matrix2.rowNum_ || matrix1.colNum_ != matrix2.colNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int pushsize = matrix1.rowNum_ * matrix2.colNum_;
	for (int i = 0;i < pushsize;i++) {
		matrix1.data[i] -= matrix2.data[i];
	}
	return matrix1;
}
Matrix_Lambda operator*(const Matrix_Lambda& matrix1, const Matrix_Lambda& matrix2) {
	if (matrix1.colNum_ != matrix2.rowNum_) {
		throw std::runtime_error("mismatched matrix");
	}
	int rows = matrix1.rowNum_, inner = matrix1.colNum_, cols = matrix2.colNum_;
	// 一次分配到位，避免 push_back 反复扩容
	std::vector<Polynomial> newdata(rows * cols, Polynomial(Complex(0, 0)));
	// ikj 顺序：i(行) → k(内维) → j(列)，B 与 C 都按行连续访问，缓存友好
	for (int i = 0; i < rows; i++) {
		int iABase = i * inner;   // 强度削减：行基址提前算好
		int iCBase = i * cols;
		for (int k = 0; k < inner; k++) {
			const Polynomial& a = matrix1.data[iABase + k];
			if (a.IsEmpty()) continue; // 跳过零多项式（IsEmpty 为精确判断，无容差）
			int kBBase = k * cols;
			for (int j = 0; j < cols; j++) {
				newdata[iCBase + j] += a * matrix2.data[kBBase + j];
			}
		}
	}
	return Matrix_Lambda(newdata, rows, cols);
}
Matrix_Lambda operator*(Polynomial pol, const Matrix_Lambda& mat) {
	std::vector<Polynomial> newdata = mat.data;
	int size = mat.rowNum_ * mat.colNum_;
	for (int i = 0;i < size;i++) {
		newdata[i] *= pol;
	}
	return Matrix_Lambda(newdata, mat.rowNum_, mat.colNum_);
}
Matrix_Lambda operator*(const Matrix_Lambda& mat, Polynomial pol) {
	return pol * mat;
}
Matrix_Lambda& operator*=(Matrix_Lambda& mat, Polynomial pol) {
	int size = mat.rowNum_ * mat.colNum_;
	for (int i = 0;i < size;i++) {
		mat.data[i] *= pol;
	}
	return mat;
}
std::ostream& operator<<(std::ostream& os, const Matrix_Lambda& mat) {
	for (int row = 0;row < mat.rowNum_;row++) {
		for (int col = 0;col < mat.colNum_;col++) {
			os << mat.data[row * mat.colNum_ + col] << "             ";
		}
		os << '\n';
	}
	return os;
}
Matrix_Complex GetComplexMatrix(const Matrix_Lambda& mat,Complex val) {
	std::vector<Complex> newdata;
	int n = mat.colNum_ * mat.rowNum_;
	newdata.resize(n, Complex(0));
	for (int i = 0;i < n;i++) {
		newdata[i] = mat.data[i].CalculatedValue(val);
	}
	return Matrix_Complex(newdata, mat.rowNum_, mat.colNum_);
}
Matrix_Lambda Hadamard(const Matrix_Lambda& mat1, const Matrix_Lambda& mat2) {
	if (mat1.colNum_ != mat2.colNum_ || mat1.rowNum_ != mat2.rowNum_) {
		throw std::runtime_error("mismatched hadamard");
	}
	std::vector<Polynomial> newdata;
	int size = mat1.colNum_ * mat1.rowNum_;
	newdata.resize(size,Polynomial({Complex(0)}));
	for (int i = 0;i < size;i++) {
		newdata[i] = mat1.data[i] * mat2.data[i];
	}
	return Matrix_Lambda(newdata, mat1.rowNum_, mat1.colNum_);
}
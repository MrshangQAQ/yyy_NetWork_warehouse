#include "Loss.h"
#include <cmath>

Matrix_Double CrossEntropyGradient(const Matrix_Double& z, const Matrix_Double& y) {
	Matrix_Double a = Softmax(z);
	return a - y;
}

std::vector<Matrix_Double> CrossEntropyGradient(const std::vector<Matrix_Double>& z,
                                                const std::vector<Matrix_Double>& y) {
	int z_row = z[0].GetRowNum();
	int z_col = z[0].GetColNum();
	Matrix_Double zt = GetFlattenMatrixFromBatch(z);
	Matrix_Double yt = GetFlattenMatrixFromBatch(y);
	Matrix_Double at_min_yt = Softmax(zt) - yt;
	return UnFlattenToBatch(at_min_yt, z_row, z_col);
}

double CrossEntropy(const Matrix_Double& z, const Matrix_Double& y, double approximation) {
	double value = 0.0;
	Matrix_Double a = Softmax(z);
	for (int i = 0; i < (int)a.GetInternalData().size(); i++) {
		if (std::abs(y.GetInternalData()[i]) < approximation) {
			continue;
		}
		value += y.GetInternalData()[i] * std::log(a.GetInternalData()[i]);
	}
	return value;
}

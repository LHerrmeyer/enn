#include <math.h>
#include "linalg.h"
#include "activ.h"

/* ReLU (rectified linear unit) */
double arelu(double x){
	return (x >= 0) ? x : 0;
}

double drelu(double output){
	return output > 0;
}

/* Leaky ReLU */
double alrelu(double x){
	return (x >= 0) ? x : 0.01*x;
}

/* Linear function is simply the identity function */
double alin(double x){
	return x;
}

/* Sigmoid activation function */
double asigm(double x){
	return 1/(1 + exp(-1*x));
}

double dsigm(double output){
	return output * (1.0 - output);
}

/* Softmax function, used for estimating probabilities from raw outputs */
Matrix* asmax(const Matrix* a){
	Matrix* out;
	int row, col;
	double sum = 0.0;

	out = mnew(a->rows, a->cols);

	/* Calculate exp(x) for each x in the matrix a, and update the sum */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] = exp(a->data[row][col]);
			sum += out->data[row][col];
		}
	}

	/* Scale each entry by the sum */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] /= sum;
		}
	}

	return out;
}

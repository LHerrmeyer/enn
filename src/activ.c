#include "activ.h"
#include "linalg.h"
#include <math.h>

Matrix* arelu(const Matrix* a){
	int i;
	Matrix *in, *out;

	/* Convert column vector to row vector so it is easier to work with */
	in = mtrns(a);

	/* Calculate the ReLU: relu(x) = 0 if x < 0, else relu(x) = x */
	for(i = 0; i < in->cols; i++){
		in->data[0][i] = (in->data[0][i] >= 0) ? in->data[0][i] : 0;
	}

	/* Transpose back to a column vector, and free the old variable */
	out = mtrns(in);
	mfree(in);

	return out;
}

/* Linear function is simply the identity function */
Matrix* alin(const Matrix* a){
	return (Matrix*)a;
}

/* Sigmoid activation function */
Matrix* asig(const Matrix* a){
	int i;
	Matrix *in, *out;

	/* Transpose matrix (convert to row vector) */
	in = mtrns(a);

	/* Calculate sigmoid */
	for(i = 0; i < in->cols; i++){
		in->data[0][i] = 1/(1 + exp(-1*in->data[0][i]));
	}

	/* Transpose vector back to column vector, free old variable*/
	out = mtrns(in);
	mfree(in);

	return out;
}

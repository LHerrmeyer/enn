#include "activ.h"
#include "linalg.h"
#include "enn.h"

Matrix* arelu(const Matrix* a){
	int i;
	Matrix *in, *out;

	/* Convert column vector to row vector so it is easier to work with */
	in = mtrans(a);

	/* Calculate the ReLU: relu(x) = 0 if x < 0, else relu(x) = x */
	for(i = 0; i < in->cols, i++){
		in[i] = (in[i] >= 0) ? in[i] : 0;
	}

	/* Transpose back to a column vector, and free the old variable */
	out = mtrans(in);
	mfree(in);

	return out;
}

/* Linear function is simply the identity function */
Matrix* alin(const Matrix* a){
	return a;
}

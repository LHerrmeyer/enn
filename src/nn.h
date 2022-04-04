#ifndef NN_H
#define NN_H
/* Data structures */
struct neural_network {
	Matrix** weights;
	Matrix** biases;
	dfunc hidden_activ; /* Input/hidden layer activation (f: double->double) */
	mfunc output_activ; /* Output layer activation (f: Matrix*->Matrix*) */
	int n_weights;
	/*int output_softmax;*/
};
typedef struct neural_network neural_network;

/* Functions */
Matrix* npred(const neural_network* nn, const Matrix* x);
#endif

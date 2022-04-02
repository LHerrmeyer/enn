#ifndef NN_H
#define NN_H
/* Data structures */
struct neural_network {
	Matrix** weights;
	Matrix** biases;
	dfunc activ_func;
	int n_weights;
};
typedef struct neural_network neural_network;

/* Functions */
Matrix* npred(neural_network* nn, const Matrix* x);
#endif

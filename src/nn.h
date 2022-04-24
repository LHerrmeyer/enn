#ifndef NN_H
#define NN_H
#include "activ.h" /* Needed for ninit() */
#include "loss.h" /* Needed for nbprop() */
/* Data structures */
struct neural_network {
	Matrix** weights;
	Matrix** biases;
	dfunc hidden_activ; /* Input/hidden layer activation (f: double->double) */
	mfunc output_activ; /* Output layer activation (f: Matrix*->Matrix*) */
	int n_layers;
};
typedef struct neural_network neural_network;

/* Functions */
Matrix* npred(const neural_network* nn, const Matrix* x);
neural_network* ninit(int inputs, int hidden_layers, int hiddens, int outputs, dfunc hidden_activ, mfunc output_activ);
Matrix*** nbprop(const neural_network* nn, const Matrix* X_train, const Matrix* y_train, const lfunc loss_func,
				const lfuncd dloss_func);
#endif

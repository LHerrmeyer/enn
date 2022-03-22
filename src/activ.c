#include "activ.h"
#include <math.h>

/* ReLU (rectified linear unit) */
double arelu(double x){
	return (x >= 0) ? x : 0;
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
double asig(double x){
	return 1/(1 + exp(-1*x));
}

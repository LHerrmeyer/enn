#include "activ.h"
#include <math.h>

/* ReLU (rectified linear unit) */
double arelu(double a){
	return a = (a >= 0) ? a : 0;
}

/* Linear function is simply the identity function */
double alin(double a){
	return a;
}

/* Sigmoid activation function */
double asig(double a){
	return 1/(1 + exp(-1*a));
}

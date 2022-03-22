#ifndef NN_H
#define NN_H
Matrix* npred(const Matrix* x, const Matrix** weights, const Matrix** biases, int n, dfunc activ_func);
#endif

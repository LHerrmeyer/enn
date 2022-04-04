#ifndef LOSS_H
#define LOSS_H
double lmse(const Matrix* actual, const Matrix* pred);
Matrix* dmse(const Matrix* actual, const Matrix* pred);
typedef double (*lfunc)(const Matrix*, const Matrix*);
typedef Matrix* (*lfuncd)(const Matrix*, const Matrix*);
#endif

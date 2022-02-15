#ifndef LINALG_H
#define LINALG_H
/* Define data structures */
/* Matrix is addressed in matrix[col][row] format to support column vectors */
struct Matrix {
	int rows;
	int cols;
	double** data; /* A 2d double array */
};
typedef struct Matrix Matrix;

/* Define function prototypes*/
Matrix* mcreate(int rows, int cols);
Matrix* mfree(Matrix* x);
Matrix* mmul(const Matrix* a, const Matrix* b);
Matrix* madd(const Matrix* a, const Matrix* b);
Matrix* mscale(const Matrix* a, int b);
void mprint(const Matrix* a);

#endif

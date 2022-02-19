#ifndef LINALG_H
#define LINALG_H
/* Define data structures */
/* Matrix is addressed in matrix[row][col] format like matrix notation and NumPy */
struct Matrix {
	int rows;
	int cols;
	double** data; /* A 2d double array */
};
typedef struct Matrix Matrix;

/* Define function prototypes*/
void mprint(const Matrix* x);
Matrix* mnew(int rows, int cols);
void mfree(Matrix* x);
Matrix* eye(int n);
Matrix* mmul(const Matrix* a, const Matrix* b);
Matrix* madd(const Matrix* a, const Matrix* b);
Matrix* mscale(const Matrix* a, int b);

#endif

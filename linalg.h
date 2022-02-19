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
Matrix* mnew(int rows, int cols);
void mfree(Matrix* x);
Matrix* mmul(const Matrix* a, const Matrix* b);
Matrix* madd(const Matrix* a, const Matrix* b);
Matrix* mscale(const Matrix* a, int b);
void mprint(const Matrix* a);

#endif

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
typedef double (*dfunc)(double);

/* Define function prototypes*/
void mprint(const Matrix* x);
Matrix* mnew(int rows, int cols);
void mfree(Matrix* x);
Matrix* mapply(const Matrix* x, dfunc func);
Matrix* meye(int n);
Matrix* mmul(const Matrix* a, const Matrix* b);
Matrix* mhad(const Matrix* a, const Matrix* b);
Matrix* madd(const Matrix* a, const Matrix* b);
Matrix* mscale(const Matrix* a, double b);
Matrix* mtrns(const Matrix* a);
int mcmp(const Matrix* a, const Matrix* b);

/* Define macros */
#define MDUP(arr,out,nrow,ncol) do { \
								Matrix* tmp; \
								int row, col; \
								tmp = mnew(nrow, ncol); \
								for(row = 0; row < nrow; row++){ \
									for(col = 0; col < ncol; col++){ \
										tmp->data[row][col] = (arr)[row][col]; \
									} \
								} \
								out = mscale(tmp, 1.0); \
								mfree(tmp); \
                               	} while (0)

#endif

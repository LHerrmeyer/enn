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
typedef Matrix* (*mfunc)(const Matrix*);

/* Define function prototypes*/
void mprint(const Matrix* x);
Matrix* mnew(int rows, int cols);
Matrix* mnew2(int rows, int cols, Matrix* a);
void mfree(Matrix* x);
Matrix* mapply(const Matrix* x, dfunc func, Matrix* out);
Matrix* meye(int n, Matrix* out);
Matrix* mconst(int rows, int cols, double value, Matrix* out);
Matrix* mmul(const Matrix* a, const Matrix* b, Matrix* out);
Matrix* mhad(const Matrix* a, const Matrix* b, Matrix* out);
Matrix* madd(const Matrix* a, const Matrix* b, Matrix* out);
Matrix* msub(const Matrix* a, const Matrix* b, Matrix* out);
Matrix* mscale(const Matrix* a, double b, Matrix* out);
Matrix* mtrns(const Matrix* a, Matrix* out);
int mcmp(const Matrix* a, const Matrix* b);
int mfrob(const Matrix* a);

/* Define macros */
#define MDUP(arr,out,nrow,ncol) do { \
								Matrix* output; \
								int row, col; \
								(out) = NULL; \
								output = (Matrix*)(out); \
								output = mnew((nrow), (ncol)); \
								for(row = 0; row < nrow; row++){ \
									for(col = 0; col < ncol; col++){ \
										output->data[row][col] = (arr)[row][col]; \
									} \
								} \
								(out) = output; \
                               	} while (0)

#endif

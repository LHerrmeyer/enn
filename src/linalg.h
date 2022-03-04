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
Matrix* meye(int n);
Matrix* mmul(const Matrix* a, const Matrix* b);
Matrix* madd(const Matrix* a, const Matrix* b);
Matrix* mscale(const Matrix* a, double b);
Matrix* mtrns(const Matrix* a);
Matrix* mdup(double*** a, int rows, int cols);
int mcmp(const Matrix* a, const Matrix* b);

/* Define macros */
#define MDUP(arr,out,nrow,ncol) do { int row, col; \
								out = mnew(nrow, ncol); \
								for(row = 0; row < nrow; row++){ \
									for(col = 0; col < ncol; col++){ \
										out->data[row][col] = arr[row][col]; \
									} \
								} \
								out = mscale(out, 1); \
                               	} while (0)

#endif

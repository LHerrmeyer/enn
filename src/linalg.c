#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "enn.h"
#include "linalg.h"

/**
* Prints out a Matrix to the screen.
*
* @param x A pointer to a Matrix to be printed
*/
void mprint(const Matrix* x){
	int col;
	int row;

	if(!x)return;

	printf("[\n");
	for(row = 0; row < x->rows; row++){
		printf("[\t");
		for(col = 0; col < x->cols; col++){
			printf("%f\t",(x->data)[row][col]);
		}
		printf("]\n");
	}
	printf("]\n");
}

/**
* Creates and allocates memory for a new Matrix.
*
* @param rows Number of rows for the matrix.
* @param cols Number of columns for the matrix.
*
* @returns A pointer to the allocated Matrix.
*/
Matrix* mnew(int rows, int cols){
	Matrix* output;
	int row;

	/* Allocate a Matrix object and set the number of rows and cols */
	output = malloc(sizeof(Matrix));
	if(!output)return NULL;
	output->rows = rows;
	output->cols = cols;

	/* The data is accessed as Matrix->data[row][col]
	Therefore, we allocate an array of rows first
	Each row is an array of ncol doubles (double*).
	*/
	output->data = malloc(rows * sizeof(double*));
	if(!output->data){
		free(output->data);
		free(output);
		return NULL;
	}
	for(row = 0; row < rows; row++){
		/* Set each row to have cols number of spots (one spot for each column in the row) */
		output->data[row] = malloc(cols * sizeof(double));
	}
	return output;
}

/**
* Creates and allocates memory for a matrix if the given matrix is not NULL.
*
* @param rows Number of rows for the Matrix.
* @param cols Number of columns for the Matrix.
* @a A pointer to an existing matrix to pass back, if NULL, a new matrix will be allocated.
*
* @returns A pointer to the allocated Matrix.
*/
Matrix* mnew2(int rows, int cols, Matrix* a){
	if(a){
		if(rows != a->rows || cols != a->cols)return NULL;
		return a;
	}
	return mnew(rows, cols);
}

/**
* Frees memory for a Matrix.
*
* @param x Pointer to a Matrix to free.
*/
void mfree(Matrix* x){
	int row;
	if(!x) return;
	for(row = 0; row < x->rows; row++){
		free(x->data[row]);
	}
	free(x->data);
	free(x);
}

/**
* Applies a function (type dfunc) to a Matrix and returns the result
*
* @param a Matrix* to apply the function to.
* @param func A function pointer (type dfunc) to apply to the Matrix.
* @param out Pointer to output matrix (optional).
*
* @returns A Matrix* of the function applied to each element in the Matrix* a.
*/

Matrix* mapply(const Matrix* a, dfunc func, Matrix* out){
	int row, col;

	if(!a || !func)return NULL;

	/* Allocate output matrix and check for NULL */
	out = mnew2(a->rows, a->cols, out);
	if(!out)return NULL;

	/* Set each cell of the output matrix to func(input matrix cell) */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] = (*func)(a->data[row][col]);
		}
	}

	return out;
}

/**
* Returns an n by n identity matrix, compare np.eye()
*
* @param n Number of rows/columns
* @param out Pointer to output matrix (optional)
*
* @returns A pointer to an n by n identity matrix
*/
Matrix* meye(int n, Matrix* out) {
	int row;
	int col;

	/* Allocate output matrix and check for null */
	out = mnew2(n, n, out);
	if(!out)return NULL;

	/* Fill with 1 for every diagonal, 0 otherwise */
	for(row = 0; row < n; row++){
		for(col = 0; col < n; col++){
			if(row == col) out->data[row][col] = 1.0;
			else out->data[row][col] = 0.0;
		}
	}

	return out;
}

/**
* Calculates a matrix full of a constant scalar value.
*
* @param rows Number of rows for the output matrix.
* @param cols Number of cols for the output matrix.
* @param num Value to fill the matrix with.
* @param out Pointer to output matrix (optional)
*
* @returns A pointer to the matrix with the scalar value.
*/
Matrix* mconst(int rows, int cols, double value, Matrix* out){
	int row;
	int col;

	/* Allocate output matrix and check for null */
	out = mnew2(rows, cols, out);
	if(!out)return NULL;

	/* Fill with constant value */
	for(row = 0; row < rows; row++){
		for(col = 0; col < cols; col++){
			out->data[row][col] = value;
		}
	}

	return out;
}
/**
* Multiply two matrices
*
* @param a Pointer to first matrix to be multiplied
* @param b Pointer to second matrix to be multiplied
* @param out Pointer to output matrix (optional)
*
* @returns A pointer to the product of the matrices
*/
Matrix* mmul(const Matrix* a, const Matrix* b, Matrix* out){
	int row, col, index;

	/* Make sure matrices are comformable and not NULL */
	if(!a || !b) return NULL;
	if(a->cols != b->rows){
		return NULL;
	}

	/* (n x m) * (m x k) -> (m x k) */
	out = mnew2(a->rows, b->cols, out);
	if(!out)return NULL;

	/* For each row in matrix a */
	for(row = 0; row < a->rows; row++){
		/* For each column matrix b */
		for(col = 0; col < b->cols; col++){
			/* Set the output cell to the sum of the products of the entries in the row of a
			and the column of b. */
			out->data[row][col] = 0;
			for(index = 0; index < a->cols; index++){
				out->data[row][col] += a->data[row][index] * b->data[index][col];
			}
		}
	}

	return out;
}

/**
* Calculates the Hadamard product of two matrices
*
* @param a Pointer to first matrix to be multiplied
* @param b Pointer to second matrix to be multiplied
* @param out Pointer to output matrix (optional)
*
* @returns A pointer to the Hadamard product of the two matrices
*/
Matrix* mhad(const Matrix* a, const Matrix* b, Matrix* out){
	int row, col;

	/* Make sure matrices have same dimensions and not NULL */
	if(!a || !b || (a->rows != b->rows) || (a->cols != b->cols)) return NULL;

	/* Allocate output matrix and check for NULL */
	out = mnew2(a->rows, a->cols, out);
	if(!out)return NULL;

	/* Calculate the Hadamard product */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] = a->data[row][col] * b->data[row][col];
		}
	}

	return out;
}

/**
* Adds two matrices together and returns the result
*
* @param a Pointer to first matrix to be added
* @param b Pointer to second matrix to be added
* @param out Pointer to output matrix (optional)
*
* @return A pointer to the matrix sum of the matrices
*/
Matrix* madd(const Matrix* a, const Matrix* b, Matrix* out){
	int row;
	int col;

	/* Make sure both have the same number of rows and columns and not NULL */
	if(!a || !b || a->rows != b->rows || a->cols != b->cols)return NULL;

	/* Allocate output matrix and check for NULL */
	out = mnew2(a->rows, a->cols, out);
	if(!out)return NULL;

	/* Set output matrix to the sum of the input matrices */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] = a->data[row][col] + b->data[row][col];
		}
	}
	return out;
}
/**
* Subtracts two matrices and returns the result
*
* @param a Pointer to first matrix to be added
* @param b Pointer to second matrix to be added
* @param out Pointer to output matrix (optional)
*
* @return A pointer to the matrix difference of the matrices
*/
Matrix* msub(const Matrix* a, const Matrix* b, Matrix* out){
	Matrix *neg_b;
	neg_b = mscale(b, -1.0, NULL);
	out = madd(a, neg_b, out);
	mfree(neg_b);
	return out;
}

/**
* Scales (multiplies) a Matrix by a scalar amount. This can also be used for duplicating a
* matrix, by setting the scalar value to 1.0
*
* @param a Pointer to the Matrix to scale.
* @param b The scalar to scale the matrix by.
* @param out Pointer to output matrix (optional)
*
* @returns The scaled Matrix.
*/
Matrix* mscale(const Matrix* a, double b, Matrix* out){
	int row, col;

	if(!a)return NULL;

	/* Allocate output matrix and check for NULL */
	out = mnew2(a->rows, a->cols, out);
	if(!out)return NULL;

	/* Set output matrix to input matrix a scaled by the scalar b */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] = a->data[row][col] * b;
		}
	}

	return out;
}

/**
* Transposes a matrix.
*
* @param a A pointer to the Matrix to transpose.
* @param out Pointer to output matrix (optional)
*
* @returns A pointer to the transposed Matrix.
*/
Matrix* mtrns(const Matrix* a, Matrix* out){
	int row, col;

	if(!a)return NULL;

	/* Allocate output matrix and check for NULL */
	out = mnew2(a->cols, a->rows, out);
	if(!out)return NULL;

	/* Set output matrix to transposed input matrix */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[col][row] = a->data[row][col];
		}
	}

	return out;
}

/**
* Calculate the Frobenius norm of a matrix
*
* @param a A pointer to the Matrix to get the norm of.
*
* @returns The norm of the Matrix.
*/
int mfrob(const Matrix* a){
	int row, col;
	double sum = 0;

	if(!a)return 0.0;

	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			sum = a->data[row][col] * a->data[row][col];
		}
	}

	return sqrt(sum);
}

/* Determine if two matrices are equal
*
* @param a First matrix to compare
* @param b Second matrix to compare
*
* @returns Whether the matrices are equal (1 or 0)
*/
int mcmp(const Matrix* a, const Matrix* b){
	int row, col;

	/* If the rows or columns are not equal, or the matrices are NULL, then return 0 */
	if(!a || !b || a->rows != b->rows || a->cols != b->cols) return 0;

	/* If any cell is not equal, then return 0 */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			if(a->data[row][col] != b->data[row][col]) return 0;
		}
	}

	/* Otherwise, if matrices are the same size and every cell is equal, return 1 */
	return 1;
}

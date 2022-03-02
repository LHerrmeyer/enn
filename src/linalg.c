#include <stdlib.h>
#include <stdio.h>
#include "linalg.h"
#include "enn.h"

/**
* Prints out a Matrix to the screen.
*
* @param x A pointer to a Matrix to be printed
*/
void mprint(const Matrix* x){
	int col;
	int row;
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
* @return A pointer to the allocated Matrix.
*/
Matrix* mnew(int rows, int cols){
	Matrix* output;
	int row;

	output = malloc(sizeof(Matrix));
	output->rows = rows;
	output->cols = cols;
	output->data = malloc(rows * sizeof(double));
	for(row = 0; row < rows; row++){
		output->data[row] = malloc(rows * sizeof(double));
	}
	return output;
}

/**
* Frees memory for a Matrix.
*
* @param x Pointer to a Matrix to free.
*/
void mfree(Matrix* x){
	int row;
	for(row = 0; row < x->rows; row++){
		free(x->data[row]);
	}
	free(x);
}

/**
* Returns an n by n identity matrix, compare np.eye()
*
* @param n Number of rows/columns
*
* @return A pointer to an n by n identity matrix
*/
Matrix* meye(int n) {
	Matrix* mat;
	int row;
	int col;

	mat = mnew(n, n);

	/* Fill with 1 for every diagonal, 0 otherwise */
	for(row = 0; row < n; row++){
		for(col = 0; col < n; col++){
			if(row == col) mat->data[row][col] = 1;
			else mat->data[row][col] = 0;
		}
	}

	return mat;
}

/**
* Adds two Matrix* together and returns the result
*
* @param a Pointer to first matrix to be added
* @param b Pointer to second matrix to be added
*
* @return A pointer to the Matrix sum of the matrices
*/
Matrix* madd(const Matrix* a, const Matrix* b){
	int row;
	int col;
	Matrix* out;

	/* Make sure both have the same number of rows and columns */
	if(a->rows != b->rows || a->cols != b->cols){
		return NULL;
	}

	/* Allocate output matrix and set it to the sum of the input matrices */
	out = mnew(a->rows, a->cols);
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] = a->data[row][col] + b->data[row][col];
		}
	}
	return out;
}

/**
* Scales (multiplies) a Matrix by a scalar amount. This can also be used for duplicating a
* matrix, by setting the scalar value to 1.0
*
* @param a Pointer to the Matrix to scale.
* @param b The scalar to scale the matrix by.
*
* @returns The scaled Matrix.
*/
Matrix* mscale(const Matrix* a, double b){
	int row, col;

	/* Allocate output matrix and set to the input matrix scaled by the scalar */
	Matrix* out = mnew(a->rows, a->cols);
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
*
* @returns A pointer to the transposed Matrix.
*/
Matrix* mtrns(const Matrix* a){
	int row, col;
	Matrix* out = mnew(a->cols, a->rows);

	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[col][row] = a->data[row][col];
		}
	}

	return out;
}

/**
* Multiplies two matrices together
*
* @param a A pointer to the 1st Matrix to be multiplied.
* @param b A pointer to the 2nd Matrix to be multiplied.
*
* @returns A pointer to the matrix product.
*/
Matrix* mmul(const Matrix* a, const Matrix* b){
	/* (m x n) * (n x k) = (m x k) */
	Matrix* product = mnew(a->rows, b->cols);
	return product;
}

/*
* Create a new matrix from a constant double**
*
* @param a The 2d array of doubles to convert to a matrix
*
* @return The duplicated matrix
*/
Matrix* mdup(double** a){
	int rows, cols;
	Matrix* mat;
	rows = sizeof(a) / sizeof(double**);
	cols = sizeof(a[0]) / sizeof(double*);

	/* Create a matrix with the constant array in it */
	mat = mnew(rows, cols);
	mat->data = a;

	/* Create a duplicate of that matrix (by multiplying by 1) */
	return mscale(mat, 1);
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

	/* If the rows or columns are not equal, then return 0 */
	if(a->rows != b->rows || a->cols != b->cols) return 0;

	/* If any cell is not equal, then return 0 */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			if(a->data[row][col] != b->data[row][col]) return 0;
		}
	}

	return 1;
}

#include <stdlib.h>
#include <stdio.h>
#include "linalg.h"
#include "enn.h"

/**
* Prints out a Matrix to the screen.
*
* @param x a pointer to a Matrix to be printed
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
* @param x Matrix to free.
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
* @return A n by n identity matrix
*/
Matrix* eye(int n) {
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
* @param a The first matrix to be added
* @param b The second matrix to be added
*
* @return The sum of the matrices
*/
Matrix* madd(const Matrix* a, const Matrix* b){
	int row;
	int col;
	Matrix* out;

	/* Make sure both have the same number of rows and columns */
	if(a->rows != b->rows || a->cols != b->cols){
		return NULL;
	}

	out = mnew(a->rows, a->cols);
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[row][col] = a->data[row][col] + b->data[row][col];
		}
	}
	return out;
}

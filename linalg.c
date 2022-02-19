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
* @return A pointer to a Matrix.
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

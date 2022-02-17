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
			printf("%f\t",(x->data)[col][row]);
		}
		printf("]\n");
	}
	printf("]\n");
}

/**
* Creates and allocates memory for a new Matrix.
*
* @param cols Number of columns for the matrix.
* @param rows Number of rows for the matrix.
* @return A pointer to a Matrix.
*/
Matrix* mnew(int cols, int rows){
	Matrix* output;
	int col;

	output = malloc(sizeof(Matrix));
	output->rows = rows;
	output->cols = cols;
	output->data = malloc(cols * sizeof(double));
	for(col = 0; col < cols; col++){
		output->data[col] = malloc(rows * sizeof(double));
	}
	return output;
}

/**
* Frees memory for a Matrix.
*
* @param x Matrix to free.
*/
void mfree(Matrix* x){
	int col;
	for(col = 0; col < x->cols; col++){
		free(x->data[col]);
	}
	free(x);
}

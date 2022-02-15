#include "linalg.h"
#include <stdlib.h>
#include <stdio.h>

void mprint(const Matrix* x){
	int col;
	int row;
	printf("[\n");
	for(col = 0; col < x->cols; col++){
		printf("[\t");
		for(row = 0; row < x->rows; row++){
			printf("%f\t",(x->data)[col][row]);
		}
		printf("]\n");
	}
	printf("]\n");
}

Matrix* mcreate(int cols, int rows){
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

void mfree(Matrix* x){
	int col;
	for(col = 0; col < x->cols; col++){
		free(x->data[col]);
	}
	free(x);
}

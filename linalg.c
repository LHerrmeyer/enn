#include "linalg.h"
#include <stdlib.h>

void mprint(Matrix* x){
	int col;
	int row;
	for(col = 0; col < x->cols, col++){
		for(row = 0; row < x->rows; row++){
			printf("%f\t",x[col][row]);
		}
	printf("\n");
	}
}

Matrix* mcreate(int rows, int cols){
	Matrix* output;
	int col;

	output = malloc(sizeof(Matrix));
	output->rows = rows;
	output->cols = cols;
	output->data = malloc(cols * sizeof(double));
	for(col = 0; col < cols; col_num++){
		output->data[col] = malloc(rows * sizeof(double));
	}
	return output;
}

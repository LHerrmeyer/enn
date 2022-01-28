#include "linalg.h"
#include <stdlib.h>

Matrix* mcreate(int rows, int cols){
	Matrix* output;
	int i;

	output = malloc(sizeof(Matrix));
	output->rows = rows;
	output->cols = cols;
	output->data = malloc(cols * sizeof(double));
	for(i = 0; i < cols; i++){
		output->data[i] = malloc(rows * sizeof(double));
	}
	return output;
}

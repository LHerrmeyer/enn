#include <stdio.h>
#include "enn.h"
#include "linalg.h"
#include "loss.h"

/**
* Calculates mean squared error of two column vectors.
*
* @param actual A column vector (Matrix*) of the actual (expected) values.
* @param pred A column vector (Matrix*) of the predicted values.
*
* @returns The mean squared error of the two column vectors.
*/
double lmse(const Matrix* actual, const Matrix* pred){
	double sum_square_error = 0, mean_square_error = 0;
	int row;

	if(actual->rows != pred->rows)return 0.0;

	for(row = 0; row < actual->rows; row++){
		sum_square_error += SQR((actual->data[row][0] - pred->data[row][0]));
	}
	mean_square_error = sum_square_error / actual->rows;

	return mean_square_error;
}

/**
* Calculates the derivative of mean squared error loss function wrt pred values
*
*/
Matrix* dmse(const Matrix* actual, const Matrix* pred){
	return msub(actual, pred, NULL);
}

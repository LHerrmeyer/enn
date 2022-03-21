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
	double sum_square_error, mean_square_error;
	int row;

	ASSERTM(actual->rows != pred->rows, "Row counts not equal");

	for(row = 0; row < actual->rows; row++){
		sum_square_error += SQR((actual->data[row][0] - pred->data[row][0]));
	}
	mean_square_error = sum_square_error / actual->rows;

	return mean_square_error;
}

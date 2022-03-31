#include <stdlib.h>
#include "enn.h"
#include "linalg.h"
#include "nn.h"

/**
* Runs the feedforward network
*
* @param x The input column vector (Matrix*) to predict on.
* @param weights An array of Matrices of the neural networks.
* @param biases An array of Matrices of the biases.
* @param n The number of weights/biases (they should be the same number)
* @param activ_func A pointer to the activation function (type dfunc)
*
* @returns A column vector of the neural network output
*/

Matrix* npred(const Matrix* x, const Matrix** weights, const Matrix** biases, int n, dfunc activ_func){
	int layer;
	Matrix *current_vector, *product, *sum;

	CHECK_NULL(x);CHECK_NULL(weights);CHECK_NULL(biases);CHECK_NULL(activ_func);

	current_vector = mscale(x, 1.0, NULL);
	for(layer = 0; layer < n; layer++){
		/* Apply the weights and biases */
		printf("Weights dimensions: %d x %d\n",
				weights[layer]->rows,
				weights[layer]->cols);
		printf("Current_vector dimensions: %d x %d\n",
				current_vector->rows,
				current_vector->cols);
		product = mmul(weights[layer], current_vector, NULL);
		CHECK_NULL(product);
		sum = madd(product, biases[layer], NULL);
		mfree(current_vector);
		mfree(product);

		/* Apply the activation function */
		current_vector = mapply(sum, activ_func, NULL);
		mfree(sum);

		/* Check for nulls */
		if(!current_vector || !sum || !product) return NULL;
	}

	/* Return the final predicted column vector */
	return current_vector;
}

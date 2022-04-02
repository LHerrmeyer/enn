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
		product = mmul(weights[layer], current_vector, NULL);
		sum = madd(product, biases[layer], NULL);
		mfree(current_vector);
		mfree(product);

		/* Apply the activation function, but not on the output layer */
		if(layer < n-1){
			current_vector = mapply(sum, activ_func, NULL);
		}
		else{
			current_vector = mscale(sum, 1.0, NULL);
		}
		mfree(sum);

		/* Check for nulls */
		/*if(!current_vector || !sum || !product) return NULL;*/
		CHECK_NULL(current_vector || sum || product);
	}

	/* Return the final predicted column vector */
	return current_vector;
}

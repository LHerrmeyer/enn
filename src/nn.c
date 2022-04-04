#include <stdlib.h>
#include "enn.h"
#include "linalg.h"
#include "nn.h"

/**
* Initializes and allocates a neural network structure
*
* @param inputs Number of input neurons
* @param hidden_layers Numbe of hidden layers
* @param hiddens Number of hidden neurons for each hidden layer
* @param outputs Numbe rof output neurons
* @param hidden_activ A pointer to an activation function for the hidden and input layers
* @param output_activ A pointer to an activation function for the output layer
*
* @returns A pointer to a neural_network structure
*/
neural_network* ninit(int inputs, int hidden_layers, int hiddens, int outputs, dfunc hidden_activ, mfunc output_activ){
	neural_network *nn;
	int i;

	/* Allocate all variables in the struct */
	nn = malloc(sizeof(neural_network));
	nn->n_weights = inputs + hidden_layers + outputs;
	nn->weights = malloc(nn->n_weights * sizeof(Matrix*));
	nn->biases = malloc(nn->n_weights * sizeof(Matrix*));
	nn->hidden_activ = hidden_activ;
	nn->output_activ = output_activ;

	/* Allocate the individual weight and bias matrices */
	/* First weight maps R^inputs -> R^hiddens, so it should be hidden rows x input cols */
	nn->weights[0] = mnew(hiddens, inputs);
	nn->biases[0] = mnew(inputs, 1);
	/* Allocate the hidden layers. These map R^hiddens -> R^hiddens*/
	for(i = 1; i < hidden_layers + 1; i++){
		nn->weights[i] = mnew(hiddens, hiddens);
		nn->biases[i] = mnew(hiddens, 1);
	}
	/* Allocate the output layer. This maps R^hiddens -> R^outputs, so it is outputs x hiddens */
	nn->weights[hidden_layers + 1] = mnew(outputs, hiddens);

	return nn;
}

/**
* Frees memory for a neural network structure
*
* @param nn A pointer to the neural network to free
*/
void nfree(neural_network* nn){
	int i;
	for(i = 0; i < nn->n_weights; i++) {
		mfree(nn->weights[i]);
		mfree(nn->biases[i]);
	}
	free(nn->weights);
	free(nn->biases);
	free(nn);
}

/**
* Runs the feedforward network
*
* @param x The input column vector (Matrix*) to predict on.
* @param nn A pointer to a neural network structure
*
* @returns A column vector of the neural network output
*/

Matrix* npred(const neural_network* nn, const Matrix* x){
	int layer;
	Matrix *current_vector, *product, *sum;

	if(!nn || !x || !nn->weights || !nn->biases)return NULL;

	current_vector = mscale(x, 1.0, NULL);
	for(layer = 0; layer < nn->n_weights; layer++){
		/* Apply the weights and biases */
		product = mmul(nn->weights[layer], current_vector, NULL);
		sum = madd(product, nn->biases[layer], NULL);
		mfree(current_vector);
		mfree(product);

		/* Apply the activation function, if it exists, but not on the output layer */
		if(nn->hidden_activ && layer < nn->n_weights-1){
			current_vector = mapply(sum, nn->hidden_activ, NULL);
		}
		else{
			current_vector = mscale(sum, 1.0, NULL);
		}
		mfree(sum);

		/* Check for nulls */
		if(!current_vector || !sum || !product) return NULL;
	}

	/* Apply output activation, if applicable */
	if(nn->output_activ){
		sum = nn->output_activ(current_vector);
		mfree(current_vector);
		current_vector = sum;
	}

	/* Return the final predicted column vector */
	return current_vector;
}

#if 1
void nbprop(neural_network* nn, Matrix* X_train, Matrix* y_train){
	size_t list_size = nn->n_weights * sizeof(Matrix*);
	/* Nabla_b and nabla_w are gradients of the biases and weights respectively. They are lists of Matrices
	just as the weights and biases are in the neural network structure */
	Matrix **nabla_b = malloc(list_size);
	Matrix **nabla_w = malloc(list_size);
	Matrix **Zs = malloc(list_size); /* A list of Z vectors (unactivated outputs) for each layer */
	Matrix **activations = malloc(list_size); /* A list of activations for each layer */
	Matrix *activation; /* Current activation */
	int layer;

	/* Set initial activation to first row of X_train */
	MDUP(&X_train->data[0], activation, 1, X_train->cols);

	/* Run the forward propagation (prediction) pass */
	for(layer = 0; layer < nn->n_weights; layer++){
		Matrix *z, *weight, *bias, *activation, *product, *sum;
		bias = nn->biases[layer];
		weight = nn->weights[layer];

		/* Calculate Z (unactivated layer output) */
		z = mmul(weight, activation, NULL);
		z = madd(z, bias, z);
		Zs[layer] = z;

		/* Calculate the activation */
	}
	return;
}
#endif

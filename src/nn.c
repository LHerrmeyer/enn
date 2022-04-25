#include <stdlib.h>
#include "enn.h"
#include "linalg.h"
#include "loss.h"
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
	if(!nn) return NULL;
	nn->n_layers = hidden_layers + 2; /* 1 Input layer + n hidden layers + 1 output layer */
	nn->weights = malloc(nn->n_layers * sizeof(Matrix*));
	nn->biases = malloc(nn->n_layers * sizeof(Matrix*));
	nn->hidden_activ = hidden_activ;
	nn->output_activ = output_activ;
	if(!nn->weights || !nn->biases){
		free(nn->weights);
		free(nn->biases);
		free(nn);
		return NULL;
	}

	/* Allocate the individual weight and bias matrices */
	/* First weight maps R^inputs -> R^hiddens, so it should be hidden rows x input cols */
	nn->weights[0] = mnew(hiddens, inputs);
	/* First biases is added after transformation to R^hiddens, so it should be in R^hiddens */
	nn->biases[0] = mnew(hiddens, 1);
	/* Allocate the hidden layers. These map R^hiddens -> R^hiddens*/
	for(i = 1; i < hidden_layers + 1; i++){
		nn->weights[i] = mnew(hiddens, hiddens);
		nn->biases[i] = mnew(hiddens, 1);
		/* Scale to 0 */
		nn->weights[i] = mscale(nn->weights[i], 0.0, nn->weights[i]);
		nn->biases[i] = mscale(nn->biases[i], 0.0, nn->biases[i]);
	}
	/* Allocate the output layer. This maps R^hiddens -> R^outputs, so it is outputs x hiddens */
	nn->weights[hidden_layers + 1] = mnew(outputs, hiddens);
	/* Last bias added to output, so it should be in R^outputs */
	nn->biases[hidden_layers + 1] = mnew(outputs, 1);

	return nn;
}

/**
* Frees memory for a neural network structure
*
* @param nn A pointer to the neural network to free
*/
void nfree(neural_network* nn){
	int i;
	for(i = 0; i < nn->n_layers; i++) {
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
	for(layer = 0; layer < nn->n_layers; layer++){
		/* Apply the weights and biases */
		product = mmul(nn->weights[layer], current_vector, NULL);
		sum = madd(product, nn->biases[layer], NULL);
		mfree(current_vector);
		mfree(product);

		/* Apply the activation function, if it exists, but not on the output layer */
		if(nn->hidden_activ && layer < nn->n_layers-1){
			current_vector = mapply(sum, nn->hidden_activ, NULL);
		}
		else{
			current_vector = mscale(sum, 1.0, NULL);
		}
		mfree(sum);
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

static Matrix* ndiff(const Matrix* x, const dfunc activ_func){
	double h = 0.000001;
	Matrix *activ_x, *activ_xh, *xh, *d_activ;

	if(!x || !activ_func) return NULL;

	/* Vector of x + h */
	xh = mconst(x->rows, x->cols, h, NULL);
	xh = madd(x, xh, xh);

	/* Numerically calculate derivative of activ_func wrt x.
	d_activ = (f(x+h)-f(x))/h */
	/* Numerator (f(x+h)-f(x)) */
	activ_x = mapply(x, activ_func, NULL);
	activ_xh = mapply(xh, activ_func, NULL);
	d_activ = msub(activ_xh, activ_x, NULL);
	/* Denominator (divide by h) */
	d_activ = mscale(d_activ, 1.0/h, d_activ);

	free(activ_xh);
	free(activ_x);
	return d_activ;
}

/**
* Run backpropagation
*
* @param nn A constant pointer to the neural network to backpropagate.
* @param X_train A pointer to the Matrix row from the training dataset to backpropagate on.
* @param y_train A pointer to the Matrix desired output for the row of the training dataset.
* @param loss_func A function pointer to the loss function.
* @param dloss_func A function pointer to the derivative of the loss function.
*
* @returns A Matrix*** of the gradients for the weights and biases. (Contains Matrix** nabla_w, Matrix** nabla_b)
*/
Matrix*** nbprop(const neural_network* nn, const Matrix* X_train, const Matrix* y_train, const lfunc loss_func,
				 const lfuncd dloss_func){
	/* http://neuralnetworksanddeeplearning.com/chap2.html#the_code_for_backpropagation */
	/* Nabla_b and nabla_w are gradients of the biases and weights respectively. They are lists of Matrices
	just as the weights and biases are in the neural network structure */
	Matrix **nabla_b, **nabla_w;
	Matrix*** nablas; /* Holds both the weight and bias gradients */
	Matrix **Zs; /* A list of Z vectors (unactivated outputs) for each layer */
	Matrix **activations = NULL; /* A list of activations for each layer */
	Matrix *z = NULL; /* Current z vector */
	Matrix *activation = NULL; /* Current activation */
	Matrix *activationp; /* Activation prime */
	Matrix *last_activation; /* Activation of last layer */
	Matrix *err; /* Error (output of loss function) */
	Matrix *delta; /* Delta for current layer */
	Matrix *tmp = NULL, *tmp2 = NULL; /* Temporary variables for calculations */
	int layer;
	size_t list_size;

	/* Check for nulls */
	if(!nn || !X_train || !y_train || !loss_func) return NULL;
	/* Make sure we only have 1 row of data */
	if(X_train->rows != 1 || y_train->rows != 1) return NULL;
	/* Make sure training data is right size */
	if(X_train->cols != nn->weights[0]->cols) return NULL;

	/* Allocate variables */
	list_size = nn->n_layers * sizeof(Matrix*);
	nabla_b = malloc(list_size);
	nabla_w = malloc(list_size);
	Zs = calloc(list_size + 1, sizeof(Matrix*));
	activations = calloc(list_size + 1, sizeof(Matrix*));

	/* Get data from designated row of X_train and y_train for stochastic gradient descent */
	/*
	MDUP(&X_train->data[index], cur_X_train, 1, X_train->cols);
	MDUP(&y_train->data[index], cur_y_train, 1, y_train->cols);
	*/

	/* Set initial activation to the row of training data (cur_X_train we are training on */
	/*activation = cur_X_train;*/
	activation = mscale(X_train, 1.0, NULL);
	activations[0] = activation;

	/* Run the forward propagation (prediction) pass */
	for(layer = 0; layer < nn->n_layers; layer++){
		Matrix *weight, *bias;
		bias = nn->biases[layer];
		weight = nn->weights[layer];

		/* Calculate Z (unactivated layer output) */
		z = mmul(weight, activation, NULL);
		z = madd(z, bias, z); /* Addition is in-place, so we don't need an extra variable */
		Zs[layer] = z;

		/* Calculate the activation */
		activation = mapply(activation, nn->hidden_activ, NULL);
		activations[layer + 1] = activation;
	}

	/* Calculate output delta*/
	delta = dloss_func(activation, y_train); /* Derviative of loss function wrt output activations */
	err = ndiff(Zs[nn->n_layers - 1], nn->hidden_activ); /* Derivative of activation function wrt output Z vector */
	delta = mhad(delta, err, NULL); /* Delta is the Hadamard product of these 2, equation BP1 */

	/* Calculate output weight and bias derivatives */
	/* Definition of dot product: x.y=x^T*y */
	last_activation = mtrns(activations[nn->n_layers - 2], NULL);
	nabla_b[nn->n_layers - 1] = mscale(delta, 1, NULL); /* Equation BP3 */
	nabla_w[nn->n_layers - 1] = mmul(delta, last_activation, NULL); /* Equation BP4 */

	mfree(err);
	mfree(tmp);
	mfree(last_activation);

	for(layer = nn->n_layers - 2; layer > 0; layer--){
		Matrix *transposed_weights, *current_weights;
		z = Zs[nn->n_layers - layer]; /* Z vector for current layer (unactivated layer output) */
		activationp = ndiff(z, nn->hidden_activ); /* Derivative of activation function for current layer */
		last_activation = mtrns(activations[layer - 1], NULL); /* Transpose of activation of layer n-1 */

		/* Calculate delta */
		transposed_weights = mtrns(nn->weights[layer + 1], NULL); /* Transposed weights of next layer */
		tmp = mmul(transposed_weights, delta, NULL);
		delta = mhad(tmp, activationp, delta); /* Equation BP2 */

		/* Calculate gradients */
		nabla_b[layer] = mscale(delta, 1.0, NULL); /* Equation BP3 */
		nabla_w[layer] = mmul(delta, last_activation, NULL); /* Equation BP4 */

		/* Free variables */
		mfree(transposed_weights);
		mfree(last_activation);
		mfree(tmp);
	}

	/* Free unneeded variables */
	for(layer = 0; layer < nn->n_layers + 1; layer++){
		mfree(Zs[layer]);
		mfree(activations[layer]);
	}
	mfree(z);
	mfree(tmp);
	mfree(tmp2);
	free(activations);
	free(Zs);

	/* Package up and return pointer to gradients */
	nablas = malloc(2 * sizeof(Matrix**));
	nablas[0] = nabla_w;
	nablas[1] = nabla_b;
	return nablas;
}

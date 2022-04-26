#include <stdlib.h>
#include "enn.h"
#include "linalg.h"
#include "loss.h"
#include "nn.h"

#ifdef NN_DBG
#define D if(1)
#define NN_PRINTF(x) do { printf x; } while (0)
#else
#define D if(0)
#define NN_PRINTF(x) if(0) x
#endif

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
	nn->weights[0] = mconst(hiddens, inputs, 1.0, NULL);
	/* First biases is added after transformation to R^hiddens, so it should be in R^hiddens */
	nn->biases[0] = mconst(hiddens, 1, 1.0, NULL);
	/* Allocate the hidden layers and initialize them with ones. These map R^hiddens -> R^hiddens*/
	for(i = 1; i < hidden_layers + 1; i++){
		nn->weights[i] = mconst(hiddens, hiddens, 1.0, NULL);
		nn->biases[i] = mconst(hiddens, 1, 1.0, NULL);
	}
	/* Allocate the output layer. This maps R^hiddens -> R^outputs, so it is outputs x hiddens */
	nn->weights[hidden_layers + 1] = mconst(outputs, hiddens, 1.0, NULL);
	/* Last bias added to output, so it should be in R^outputs */
	nn->biases[hidden_layers + 1] = mconst(outputs, 1, 1.0, NULL);

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
	int layer, layer_fwd;
	size_t list_size;

	/* Check for nulls */
	if(!nn || !X_train || !y_train || !loss_func) return NULL;
	/* Make sure we only have 1 row of data */
	if(X_train->rows != 1 || y_train->rows != 1) return NULL;
	/* Make sure training data is right size */
	if(X_train->cols != nn->weights[0]->cols) return NULL;

	/* Allocate variables */
	list_size = nn->n_layers * sizeof(Matrix*);
	/* nabla_b = [np.zeros(b.shape) for b in self.biases] */
	nabla_b = malloc(list_size);
	/* nabla_w = [np.zeros(w.shape) for w in self.weights] */
	nabla_w = malloc(list_size);
	/* zs = [] */
	Zs = calloc(list_size + 1, sizeof(Matrix*));
	/* activations = [x] */
	activations = calloc(list_size + 1, sizeof(Matrix*));

	/* Get data from designated row of X_train and y_train for stochastic gradient descent */
	/*
	MDUP(&X_train->data[index], cur_X_train, 1, X_train->cols);
	MDUP(&y_train->data[index], cur_y_train, 1, y_train->cols);
	*/

	/* Set initial activation to the row of training data (cur_X_train we are training on */
	/* activation = x */
	activation = mscale(X_train, 1.0, NULL);
	activations[0] = activation;

	/* Run the forward propagation (prediction) pass */
	/* for b, w in zip(self.biases, self.weights): */
	for(layer = 0; layer < nn->n_layers; layer++){
		Matrix *weight, *bias;
		bias = nn->biases[layer];
		weight = nn->weights[layer];

		/* Calculate Z (unactivated layer output) */
		/* z = np.dot(w, activation)+b */
		z = mmul(weight, activation, NULL);
		z = madd(z, bias, z); /* Addition is in-place, so we don't need an extra variable */
		/* zs.append(z)  */
		Zs[layer] = z;

		/* Calculate the activation by applying it to Z (the output of the layer before activtion) */
		/* activation = sigmoid(z) */
		activation = mapply(z, nn->hidden_activ, NULL);
		/* activations.append(activation) */
		activations[layer + 1] = activation;
	}

	/* Calculate output delta*/
	/* delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) */
	delta = dloss_func(activation, y_train); /* Derviative of loss function wrt output activations */
	D printf("Last activation:\n");
	D mprint(activation);
	D printf("y_train set:\n");
	D mprint(y_train);
	D printf("Derivative of loss func:\n");
	D mprint(delta);
	err = ndiff(Zs[nn->n_layers - 1], nn->hidden_activ); /* Derivative of activation function wrt output Z vector */
	D printf("Derivative of activation function wrt output Z vector:\n");
	D mprint(err);
	delta = mhad(delta, err, NULL); /* Delta is the Hadamard product of these 2, equation BP1 */
	D printf("Delta (Hadamard product):\n");
	D mprint(delta);

	/* Calculate output weight and bias derivatives */
	/* Definition of dot product: x.y=x^T*y */
	last_activation = mtrns(activations[nn->n_layers - 2], NULL);
	/* nabla_b[-1] = delta */
	/* In a 4 layer network, this would be nabla_b[3] */
	/* Last element of nabla_b is nn->n_layers - 1 and not nn->n_layers */
	nabla_b[nn->n_layers - 1] = mscale(delta, 1, NULL); /* Equation BP3 */
	/* nabla_w[-1] = np.dot(delta, activations[-2].transpose()) */
	nabla_w[nn->n_layers - 1] = mmul(delta, last_activation, NULL); /* Equation BP4 */

	mfree(err);
	mfree(tmp);
	mfree(last_activation);

	/*  for l in xrange(2, self.num_layers): */
	/* In 4 layer network:
	weights[0]
	weights[1]
	weights[2]
	weights[3]
	nn->n_layers = 4
	weights[-1] should be weights[3]
	weights[-2] should be weights[2]
	So, the layer_fwd should go 2, 3 which corresponds to layers 2, 1
	*/
	D printf("Starting backward pass\n");
	for(layer_fwd = 2; layer_fwd < nn->n_layers; layer_fwd++){
		int layer = nn->n_layers - layer_fwd;
		Matrix *transposed_weights, *current_weights;
		D printf("=====================New loop iteration=================\n");
		D printf("Layer: %d, layer_fwd: %d\n",layer,layer_fwd);
		D printf("Next layer's (%d) weights:\n", layer+1);
		D mprint(nn->weights[layer + 1]);
		D printf("This layer's (%d) weights:\n", layer);
		D mprint(nn->weights[layer]);
		/*  z = zs[-l] */
		z = Zs[layer]; /* Z vector for current layer (unactivated layer output) */
		/* sp = sigmoid_prime(z) */
		activationp = ndiff(z, nn->hidden_activ); /* Derivative of activation function for current layer */
		/* last_activation = activations[-l-1].transpose() */
		last_activation = mtrns(activations[layer - 1], NULL); /* Transpose of activation of layer n-1 */
		D printf("z:\n");
		D mprint(z);
		D printf("activationp:\n");
		D mprint(activationp);
		D printf("last_activation:\n");
		D mprint(last_activation);

		/* Calculate delta */
		/* delta = np.dot(self.weights[-l+1].transpose(), delta) * sp */
		/* tmp = np.dot(self.weights[-l+1].transpose(), delta) */
		transposed_weights = mtrns(nn->weights[layer + 1], NULL); /* Transposed weights of next layer */
		tmp = mmul(transposed_weights, delta, NULL);
		/* delta = tmp * sp */
		D printf("transposed_weights:\n");
		D mprint(transposed_weights);
		D printf("tmp (transposed_weights @ delta):\n");
		D mprint(tmp);
		D printf("delta:\n");
		D mprint(delta);
		delta = mhad(tmp, activationp, NULL); /* Equation BP2 */
		D printf("New delta (tmp * activationp):\n");
		D mprint(delta);

		/* Calculate gradients */
		/* nabla_b[-l] = delta */
		nabla_b[layer] = mscale(delta, 1.0, NULL); /* Equation BP3 */
		D printf("nabla_b (delta):\n");
		D mprint(nabla_b[layer]);
		/* nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) */
		nabla_w[layer] = mmul(delta, last_activation, NULL); /* Equation BP4 */
		D printf("nabla_w ( np.dot(delta, activations[-l-1].transpose()) ) :\n");
		D mprint(nabla_w[layer]);
		D printf("Last activation:\n");
		D mprint(last_activation);
		D printf("activations[layer-1]:\n");
		D mprint(activations[layer-1]);
		D printf("Weight dimensions: %d x %d, nabla_w dimensions: %d x %d\n",
					nn->weights[layer]->rows, nn->weights[layer]->cols,
					nabla_w[layer]->rows, nabla_w[layer]->cols);

		/* Free variables */
		mfree(transposed_weights);
		mfree(last_activation);
		mfree(tmp);
	}


	/* Free unneeded variables */
	for(layer = 0; layer < nn->n_layers + 1; layer++){
		mfree(Zs[layer]);
		D printf("==================================================\n");
		D printf("Activations[%d]\n",layer);
		D mprint(activations[layer]);
		D printf("Nabla_w[%d]\n",layer);
		D mprint(nabla_w[layer]);
		D printf("nn->weights[%d]\n",layer);
		D mprint(nn->weights[layer]);
		mfree(activations[layer]);
	}
	mfree(tmp2);
	free(activations);
	free(Zs);

	/* Package up and return pointer to gradients */
	nablas = malloc(2 * sizeof(Matrix**));
	nablas[0] = nabla_w;
	nablas[1] = nabla_b;
	return nablas;
}

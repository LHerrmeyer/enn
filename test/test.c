#include <stdio.h>
#include <stdlib.h>
#include "../src/enn.h"
#include "../src/linalg.h"
#include "../src/activ.h"
#include "../src/nn.h"
#include "../src/loss.h"
#include "minunit.h"

int tests_run = 0;

static char* test_mnew(){
	Matrix* mat;
	mat = mnew(5,3);
	mu_assert("Error, rows != 5", mat->rows == 5);
	mu_assert("Error, cols != 3", mat->cols == 3);
	mfree(mat);
	return NULL;
}

static char* test_mnew2(){
	Matrix *mat, *mat2;
	mat = mnew2(5, 3, NULL);
	mat2 = mnew2(5, 3, mat);
	mu_assert("Error, mat is NULL", mat);
	mu_assert("Error, mat != mat2", mat == mat2);
	mfree(mat); /* Only one needs to be freed as mat and mat2 point to the same memory */
	return NULL;
}

static char* test_mcmp(){
	Matrix *a, *a2, *b, *b2;
	double a_data[2][2] = {
		{1.0, 0.0},
		{0.0, 1.0}
	};
	double b_data[2][3] = {
		{5.0, 4.0, 3.0},
		{2.0, 1.0, 0.0}
	};

	/* Convert arrays in to matrices */
	MDUP(a_data, a, 2, 2);
	MDUP(a_data, a2, 2, 2);
	MDUP(b_data, b, 2, 3);
	MDUP(b_data, b2, 2, 3);

	/* Run the tests */
	mu_assert("Error, a != a2", mcmp(a, a2));
	mu_assert("Error, a == b", !mcmp(a, b));
	mu_assert("Error, b != b2", mcmp(b, b2));

	/* Free matrix variables */
	mfree(a);
	mfree(a2);
	mfree(b);
	mfree(b2);

	return NULL;
}

static char* test_mconst(){
	Matrix *a, *b, *a2, *b2;
	double a_data[2][2] = {
		{1.0, 1.0},
		{1.0, 1.0}
	};
	double b_data[2][3] = {
		{2.0, 2.0, 2.0},
		{2.0, 2.0, 2.0}
	};

	/* Initialize matrices */
	MDUP(a_data, a, 2, 2);
	MDUP(b_data, b, 2, 3);
	a2 = mconst(2, 2, 1.0, NULL);
	b2 = mconst(2, 3, 2.0, NULL);

	/* Run the tests */
	mu_assert("Error, a != a2", mcmp(a, a2));
	mu_assert("Error, b != b2", mcmp(b, b2));

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(a2);
	mfree(b2);

	return NULL;
}

static char* test_mmul(){
	Matrix *a, *b, *c, *prod;
	double ad[2][2] = {
		{1., 2.},
		{2., 1.}
	};
	double bd[2][3] = {
		{2., 3., 4.},
		{5., 6., 7.}
	};
	double cd[2][3] = {
		{12., 15., 18.},
		{9., 12., 15.}
	};

	/* Convert arrays into matrices */
	MDUP(ad, a, 2, 2);
	MDUP(bd, b, 2, 3);
	MDUP(cd, c, 2, 3);

	/* Run tests */
	prod = mmul(a, b, NULL);
	mu_assert("Error, a * b != c", mcmp(prod, c));
	mfree(prod);
	prod = mmul(b, a, NULL);
	mu_assert("Error, b * a != NULL", !prod);
	mfree(prod);

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(c);

	return NULL;
}

static char* test_mhad(){
	Matrix *a, *b, *c, *prod;
	double ad[2][2] = {
		{1., 2.},
		{3., 4.}
	};
	double bd[2][2] = {
		{2., 1.},
		{1., 2.}
	};
	double cd[2][2] = {
		{2., 2.},
		{3., 8.}
	};

	/* Convert arrays into matrices */
	MDUP(ad, a, 2, 2);
	MDUP(bd, b, 2, 2);
	MDUP(cd, c, 2, 2);

	/* Run tests */
	prod = mhad(a, b, NULL);
	mu_assert("Error, mhad(a, b) != c", mcmp(prod, c));

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(c);
	mfree(prod);

	return NULL;
}

static char* test_mscale(){
	Matrix *a, *as, *b, *c, *cs, *d;
	double ad[2][2] = {
		{1.0, 2.0},
		{3.0, 4.0}
	};
	double bd[2][2] = {
		{2.0, 4.0},
		{6.0, 8.0}
	};

	MDUP(ad, a, 2, 2);
	MDUP(bd, b, 2, 2);

	c = mnew(1, 1);
	d = mnew(1, 1);
	c->data[0][0] = -8.0;
	d->data[0][0] = -8.0;

	as = mscale(a, 2.0, NULL);
	mu_assert("Error, b != mscale(a)", mcmp(b, as));
	cs = mscale(c, 1.0, NULL);
	mu_assert("Error, d != mscale(c)", mcmp(d, cs));

	mfree(as);
	mfree(a);
	mfree(b);

	return NULL;
}

/* This function tests both arelu() and mapply() */
static char* test_arelu(){
	Matrix *a_in, *a_int, *a_out, *a_outt, *res;
	double a_ind[6] = {1.0, 5.0, 4.0, 8.0, -5.0, -1.0};
	double a_outd[6] = {1.0, 5.0, 4.0, 8.0, 0.0, 0.0};

	/* Convert arrays to matrices */
	MDUP((&a_ind), a_in, 1, 6);
	MDUP((&a_outd), a_out, 1, 6);

	/* Transpose (convert to column vectors) */
	a_int = mtrns(a_in, NULL);
	a_outt = mtrns(a_out, NULL);

	/* Run tests*/
	res = mapply(a_int, &arelu, NULL);
	mu_assert("Error, arelu(a_in) != a_out", mcmp(res, a_outt));

	/* Free variables */
	mfree(a_in);
	mfree(a_out);
	mfree(a_int);
	mfree(a_outt);
	mfree(res);

	return NULL;
}

static char* test_npred(){
	#define N_TESTS 10
	Matrix **weights, **biases, *out_prob, *current_vector, *current_vector_trns;
	neural_network* nn;
	int i, j, n_layers, prediction;
	double pred_max;
	/* Weights and biases of neural network to test predictions */
	double w0[4][4] = {
		{-0.5206975 ,  0.5338802 , -0.5602411 , -0.09294045},
		{-0.81646407,  0.07859222,  0.8910857 ,  0.9753645 },
		{ 0.0776132 , -0.71796286,  1.0895936 ,  0.40837875},
		{-0.46662232,  0.19200796,  0.38742024, -0.2863772 }
	};
	double w1[4][4] = {
		{ 0.2019741 , -0.71195257, -0.8410556 ,  0.6462495 },
		{-0.636823  ,  1.4791069 ,  0.25363532, -0.30699533},
		{-0.7421215 ,  1.5144516 ,  0.48467913,  0.81691414},
		{-0.62316686, -0.7518175 ,  0.7958357 , -0.5908574 }
	};
	double w2[3][4] = {
		{ 0.24331057, -1.0454109 , -1.8839567 , -1.2707748 },
		{-0.88661844, -1.3611857 ,  0.29023024,  1.1938326 },
		{ 0.01811641,  0.8420355 ,  0.980748  , -0.07365165}
	};
	double b0[4] = {0.0, -0.563959, -0.06092859, 0.0};
	double b1[4] = {0.0, -0.82546085, -0.3782354, -0.00169147};
	double b2[3] = {1.9372896, -0.7055002, -1.4840443};
	/*double test_set_X[5][4] = {
		{6.1, 2.8, 4.7, 1.2},
		{5.7, 3.8, 1.7, 0.3},
		{7.7, 2.6, 6.9, 2.3},
		{6. , 2.9, 4.5, 1.5},
		{6.8, 2.8, 4.8, 1.4}
	};
	int test_set_y[5] = {1, 0, 2, 1, 1};
	*/
	double test_set_X[N_TESTS][4] = {
		{6.1, 2.8, 4.7, 1.2},
		{5.7, 3.8, 1.7, 0.3},
		{7.7, 2.6, 6.9, 2.3},
		{6. , 2.9, 4.5, 1.5},
		{6.8, 2.8, 4.8, 1.4},
		{5.4, 3.4, 1.5, 0.4},
		{5.6, 2.9, 3.6, 1.3},
		{6.9, 3.1, 5.1, 2.3},
		{6.2, 2.2, 4.5, 1.5},
		{5.8, 2.7, 3.9, 1.2}
	};
	int test_set_y[N_TESTS] = {1, 0, 2, 1, 1, 0, 1, 2, 1, 1};
	n_layers = 4;

	/* Allocate variables for weights and biases */
	weights = malloc(sizeof(Matrix*) * (n_layers - 1));
	biases = malloc(sizeof(Matrix*) * (n_layers - 1));

	/* Put weights and biases into Matrix* structs */
	MDUP(w0, weights[0], 4, 4);
	MDUP(w1, weights[1], 4, 4);
	MDUP(w2, weights[2], 3, 4);
	MDUP(&b0, biases[0], 1, 4);
	MDUP(&b1, biases[1], 1, 4);
	MDUP(&b2, biases[2], 1, 3);

	/* Convert biases to column vectors */
	for(i = 0; i < n_layers - 1; i++){
		Matrix* cur_bias = biases[i];
		biases[i] = mtrns(cur_bias, NULL);
		mfree(cur_bias);
	}

	/* Put everything into a neural network object */
	nn = malloc(sizeof(neural_network));
	nn->weights = weights;
	nn->biases = biases;
	nn->hidden_activ = &arelu;
	nn->output_activ = &asmax;
	nn->n_layers = n_layers;

	/* Run the tests */
	for(i = 0; i < N_TESTS; i++){
		/* Run the neural network prediction */
		MDUP(&test_set_X[i], current_vector_trns, 1, 4);
		current_vector = mtrns(current_vector_trns, NULL);
		out_prob = npred(nn, current_vector); /* Prediction is a probably as we are using softmax output */

		/* Find the prediction using argmax */
		pred_max = 0.0;
		prediction = 0;
		for(j = 0; j < 3; j++){
			if(out_prob->data[j][0] > pred_max){
				pred_max = out_prob->data[j][0];
				prediction = j;
			}
		}

		/* Test against actual TensorFlow predictions */
		mu_assert("Error: prediction != actual", prediction == test_set_y[i]);

		/* Free variables */
		mfree(current_vector_trns);
		mfree(current_vector);
		mfree(out_prob);
	}

	/* Free the rest of the variables */
	mfree(weights[0]);
	mfree(weights[1]);
	mfree(weights[2]);
	mfree(biases[0]);
	mfree(biases[1]);
	mfree(biases[2]);
	free(weights);
	free(biases);
	free(nn);

	return NULL;
}

static char* test_mfree(){
	Matrix* abc;
	abc = mnew(10,20);
	mfree(abc);
	return NULL;
}

static char* test_nbprop(){
	Matrix *test_X2, *test_y2, *test_X, *test_y;
	Matrix *cur_X, *cur_y, *preds, *pred, *x_in;
	Matrix ***gradients, **weight_gradients, **bias_gradients;
	/* 1 input neuron on input layer, 2 hidden layers /w 2 neurons each, 1 output layer with 1 neuron */
	/* Weight 1 maps R^1 -> R^2 (2 x 1) layer 0 (input) -> layer 1
	Layer 1 maps R^2 -> R^2 (2 x 2) layer 1 -> layer 2
	Layer 3 maps R^2 -> R^1 (1 x 2) layer 2 -> layer 3 (output)
	There are supposed to be n-1 weights and not n weights, why?
	Ah, weights are between layers and not on them. The last doesn't need a weight.
	Weights correspond to a transformation between layers, not a layer itself.
	*/
	neural_network* nn = ninit(1, 2, 2, 1, &alrelu, NULL);
	int current_index = 0, i, epoch;
	double mse;
	double learning_rate = 0.0025;
	/* Data is from Anscombe's quartet set 1. Equation should be y=3x+5, and MSE for a linear model is 1.25 */
	double test_set_X[11] = {10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0};
	double test_set_y[11] = {8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68};

	/* Convert test and training sets to matrices and convert them to column vectors */
	MDUP(&test_set_X, test_y2, 1, 11); /* 1 row, 11 cols */
	MDUP(&test_set_y, test_X2, 1, 11);
	test_X = mtrns(test_X2, NULL); /* Converting into 11 rows, 1 col */
	test_y = mtrns(test_y2, NULL);
	mfree(test_X2);
	mfree(test_y2);

	for(epoch = 0; epoch < 2000; epoch++){
		int current_index = epoch % test_X->rows;
		printf("Current epoch: %d, index: %d\n", epoch, current_index);
		/* Get current X and y values to train on */
		MDUP(&test_X->data[current_index], cur_X, 1, 1); /* One row, one col */
		MDUP(&test_y->data[current_index], cur_y, 1, 1);
		printf("Current x: %f\n",cur_X->data[0][0]);
		printf("Current y: %f\n",cur_y->data[0][0]);

		/* Start backprop with mean squared error loss function */
		gradients = nbprop(nn, cur_X, cur_y, lmse, dmse);
		weight_gradients = gradients[0];
		bias_gradients = gradients[1];

		printf("Updating weights...\n");
		for(i = 0; i < nn->n_layers - 1; i++){
			Matrix *cur_weight_gradient = mscale(weight_gradients[i], learning_rate, NULL);
			Matrix *cur_bias_gradient = mscale(bias_gradients[i], learning_rate, NULL);
			/*printf("Current weight gradient:\n");
			mprint(weight_gradients[i]);
			printf("Current bias gradient:\n");
			mprint(bias_gradients[i]);*/
			nn->weights[i] = msub(nn->weights[i], cur_weight_gradient, nn->weights[i]);
			nn->biases[i] = msub(nn->biases[i], cur_bias_gradient, nn->biases[i]);
			mfree(cur_weight_gradient);
			mfree(cur_bias_gradient);
			printf("Current weights:\n");
			mprint(nn->weights[i]);
			printf("Current biases:\n");
			mprint(nn->biases[i]);
		}
		/* Calculate MSE */
		preds = mnew(test_X->rows, 1);
		for(i = 0; i < test_X->rows; i++){
			MDUP(test_X->data + i, x_in, 1, 1);
			/*printf("X_in:\n");
			mprint(x_in);*/
			/*printf("nn->weights[0]:\n");
			mprint(nn->weights[0]);*/
			pred = npred(nn, x_in);
			/*printf("pred:\n");
			mprint(pred);*/
			preds->data[i][0] = pred->data[0][0];
			mfree(pred);
		}
		/*
		printf("Preds:\n");
		mprint(preds);
		printf("Actuals:\n");
		mprint(test_y);*/
		mse = lmse(test_y, preds);
		printf("mse: %f\n",mse);
	}
	
	return NULL;
}

static char* all_tests(){;
	mu_run_test(test_mnew);
	mu_run_test(test_mnew2);
	mu_run_test(test_mcmp);
	mu_run_test(test_mconst);
	mu_run_test(test_mmul);
	mu_run_test(test_mhad);
	mu_run_test(test_mscale);
	mu_run_test(test_arelu);
	mu_run_test(test_npred);
	mu_run_test(test_mfree);
	mu_run_test(test_nbprop);
	return NULL;
}

int main(void){
	char* result = all_tests();
	if(result != NULL){
		printf("%s\n",result);
	}
	else{
		printf("All tests passed!\n");
	}
	printf("Tests run: %d\n", tests_run);

	return result != NULL;
}

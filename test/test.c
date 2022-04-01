#include <stdio.h>
#include <stdlib.h>
#include "../src/enn.h"
#include "../src/linalg.h"
#include "../src/activ.h"
#include "../src/nn.h"
#include "minunit.h"

int tests_run = 0;

static char* test_mnew(){
	Matrix* mat;
	mat = mnew(5,3);
	mu_assert("Error, rows != 5", mat->rows == 5);
	mu_assert("Error, cols != 3", mat->cols == 3);
	return NULL;
}

static char* test_mnew2(){
	Matrix *mat, *mat2;
	mat = mnew2(5, 3, NULL);
	mat2 = mnew2(5, 3, mat);
	mu_assert("Error, mat is NULL", mat);
	mu_assert("Error, mat != mat2", mat == mat2);
	return NULL;
}

static char* test_meye(){
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
	Matrix **weights, **biases, *out, *out_prob, *current_vector, *current_vector_trns;
	int i, j, n_tests, n_layers, prediction;
	double pred_max;
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
	double test_set_X[5][4] = {
		{6.1, 2.8, 4.7, 1.2},
		{5.7, 3.8, 1.7, 0.3},
		{7.7, 2.6, 6.9, 2.3},
		{6. , 2.9, 4.5, 1.5},
		{6.8, 2.8, 4.8, 1.4}
	};
	int test_set_y[5] = {1, 0, 2, 1, 1};
	n_tests = 5;
	n_layers = 3;

	/* Allocate variables for weights and biases */
	weights = malloc(sizeof(Matrix*) * n_layers);
	biases = malloc(sizeof(Matrix*) * n_layers);

	/* Put weights and biases into Matrix* structs */
	MDUP(w0, weights[0], 4, 4);
	MDUP(w1, weights[1], 4, 4);
	MDUP(w2, weights[2], 3, 4);
	MDUP(&b0, biases[0], 1, 4);
	MDUP(&b1, biases[1], 1, 4);
	MDUP(&b2, biases[2], 1, 3);

	/* Convert biases to column vectors */
	for(i = 0; i < n_layers; i++){
		Matrix* cur_bias = biases[i];
		biases[i] = mtrns(cur_bias, NULL);
		mfree(cur_bias);
	}

	/* Run the tests */
	for(i = 0; i < n_tests; i++){
		/* Run the neural network prediction */
		MDUP(&test_set_X[i], current_vector_trns, 1, 4);
		current_vector = mtrns(current_vector_trns, NULL);
		out = npred(current_vector, (const Matrix**)weights, (const Matrix**)biases, n_layers, &arelu);

		/* Find the prediction using argmax */
		out_prob = asmax(out);
		pred_max = 0;
		prediction = 0;
		for(j = 0; j < 3; j++){
			if(out_prob->data[j][0] > pred_max) prediction = j;
		}

		/* Test against actual TensorFlow predictions */
		if(prediction != test_set_y[i]){
			printf("Prediction: %d, Actual: %d, Index: %d\n", prediction, test_set_y[i], i);
			mprint(out_prob);
		}
		mu_assert("Error: prediction != actual", prediction == test_set_y[i]);
		mprint(out_prob);

		/* Free variables */
		mfree(current_vector_trns);
		mfree(current_vector);
		mfree(out);
	}
	return NULL;
}

static char* all_tests(){;
	mu_run_test(test_mnew);
	mu_run_test(test_mnew2);
	mu_run_test(test_meye);
	mu_run_test(test_mmul);
	mu_run_test(test_mhad);
	mu_run_test(test_arelu);
	mu_run_test(test_npred);
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

#include <stdio.h>
#include <stdlib.h>
#include "../src/enn.h"
#include "../src/linalg.h"
#include "minunit.h"

int tests_run = 0;

static char* test_mnew(){
	Matrix* mat;
	mat = mnew(5,3);
	mu_assert("Error, rows != 5", mat->rows == 5);
	mu_assert("Error, cols != 3", mat->cols == 3);
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
	prod = mmul(a, b);
	mu_assert("Error, a * b != c", mcmp(prod, c));
	mfree(prod);
	prod = mmul(b, a);
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
	prod = mhad(a, b);
	mu_assert("Error, mhad(a, b) != c", mcmp(prod, c));
	mfree(prod);

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(c);

	return NULL;
}

static char* all_tests(){;
	mu_run_test(test_mnew);
	mu_run_test(test_meye);
	mu_run_test(test_mmul);
	mu_run_test(test_mhad);
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

#include <stdio.h>
#include <stdlib.h>
#include "../src/enn.h"
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
	free(a);
	free(a2);
	free(b);
	free(b2);

	return NULL;
}

static char* all_tests(){;
	mu_run_test(test_mnew);
	mu_run_test(test_meye);
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

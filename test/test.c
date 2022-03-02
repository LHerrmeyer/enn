#include <stdio.h>
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
	int res;
	double i3_data[3][3] = {
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0}
	};

	res = mcmp(mdup((double**)i3_data), meye(3));
	mu_assert("Error meye(3) != mdup(i3_data)", res);
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

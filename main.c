#include <stdio.h>
#include "linalg.h"

int main(void){
	Matrix* mat = mcreate(2,2); /* 2 cols, 2 rows */
	mat->data[0][0] = 4.0;
	mat->data[0][1] = 5.0;
	mat->data[1][0] = 6.0;
	mat->data[1][1] = 7.0;
	mprint(mat);
	return 0;
}

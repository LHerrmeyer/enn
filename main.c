#include <stdio.h>
#include "enn.h"
#include "linalg.h"

int main(void){
	Matrix* mat = mnew(2,2); /* 2 rows, 2 cols */
	/* Row 1 */
	mat->data[0][0] = 4.0;
	mat->data[0][1] = 5.0;
	/* Row 2 */
	mat->data[1][0] = 6.0;
	mat->data[1][1] = 7.0;
	mprint(mat);
	mfree(mat);
	return 0;
}

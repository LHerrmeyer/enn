#include <stdio.h>
#include "enn.h"
#include "linalg.h"

int main(void){
	Matrix *mat, *i_mat, *sum;
	mat = mnew(2,2); /* 2 rows, 2 cols */
	/* Row 1 */
	mat->data[0][0] = 4.0;
	mat->data[0][1] = 5.0;
	/* Row 2 */
	mat->data[1][0] = 6.0;
	mat->data[1][1] = 7.0;
	mprint(mat);

	i_mat = meye(2, NULL);
	mprint(i_mat);

	sum = madd(mat, i_mat, NULL);
	mprint(sum);

	mfree(mat);
	mfree(i_mat);
	mfree(sum);
	return 0;
}

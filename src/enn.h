#ifndef ENN_H
#define ENN_H
#include <stdio.h>
#include <stdlib.h>
/* Source https://ocw.cs.pub.ro/courses/so/laboratoare/resurse/die */
#define ASSERTM(assertion, msg)				\
	do {									\
		if (assertion) {					\
			fprintf(stderr, "(%s, %d): ", 	\
					__FILE__, __LINE__); 	\
			fprintf(stderr, msg);			\
			exit(-1);						\
		}									\
	} while (0)
#define SQR(x) ((x)*(x))
#endif

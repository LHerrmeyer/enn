#ifndef ENN_H
#define ENN_H
#include <stdio.h>
#include <stdlib.h>
static const char* check_if_fmt = "**** Check <%s> failed at %s:%d ****\n";
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
#define CHECK_IF(expr)						\
	do {									\
		if(expr){							\
			fprintf(stderr, check_if_fmt,	\
					#expr, __FILE__,		\
					__LINE__);				\
		}									\
	} while(0)
#define CHECK_NULL(expr) CHECK_IF(!expr)
#define SQR(x) ((x)*(x))
#endif

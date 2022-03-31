#ifndef ENN_H
#define ENN_H
#include <stdio.h>
#include <stdlib.h>
#ifdef __GNUC__
#define UNUSED_VAR __attribute__ ((unused))
#else
#define UNUSED_VAR
#endif
static const char* UNUSED_VAR check_if_fmt = "**** Assertion <%s> failed at %s:%d ****\n";
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
#define ENN_ASSERT(expr) ((expr) != 0) ?0 :fprintf(stderr, check_if_fmt, (#expr), __FILE__, __LINE__)
#define CHECK_NULL ENN_ASSERT
#define ENN_ASSERT_F(expr) ENN_ASSERT(!(expr))
#define SQR(x) ((x)*(x))
#endif

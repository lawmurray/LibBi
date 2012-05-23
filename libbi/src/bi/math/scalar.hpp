/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_SCALAR_HPP
#define BI_MATH_SCALAR_HPP

#include "../cuda/cuda.hpp"

/**
 * Value type for calculations, float or double. We use this rather
 * than templates because of limited function template support on the GPU
 * and necessary use of global variables.
 */
#ifdef ENABLE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

/**
 * @def BI_REAL
 *
 * Cast alias.
 */
#define BI_REAL(x, ...) static_cast<real>(x,##__VA_ARGS__)

/**
 * @def BI_IS_FINITE
 *
 * Is value finite?
 */
#define BI_IS_FINITE(x) ((x)*0 == 0)

/*
 * Function aliases.
 */
#ifdef ENABLE_DOUBLE
#define BI_MATH_FABS(x) fabs(x)
#define BI_MATH_LOG(x) log(x)
#define BI_MATH_NANLOG(x) (isnan(x) ? log(0.0) : log(x))
#define BI_MATH_EXP(x) exp(x)
#define BI_MATH_NANEXP(x) (isnan(x) ? 0.0 : exp(x))
#define BI_MATH_MAX(x,y) fmax(x,y)
#define BI_MATH_MIN(x,y) fmin(x,y)
#define BI_MATH_SQRT(x) sqrt(x)
#define BI_MATH_POW(x,y) pow(x,y)
#define BI_MATH_FMOD(x,y) fmod(x,y)
#define BI_MATH_MODF(x,y) modf(x,y)
#define BI_MATH_CEIL(x) ceil(x)
#define BI_MATH_FLOOR(x) floor(x)
#define BI_MATH_TGAMMA(x) tgamma(x)
#define BI_MATH_LGAMMA(x) lgamma(x)
#define BI_MATH_SIN(x) sin(x)
#define BI_MATH_COS(x) cos(x)
#define BI_MATH_TAN(x) tan(x)
#define BI_MATH_ASIN(x) asin(x)
#define BI_MATH_ACOS(x) acos(x)
#define BI_MATH_ATAN(x) atan(x)
#define BI_MATH_ATAN2(x,y) atan2(x,y)
#define BI_MATH_SINH(x) sinh(x)
#define BI_MATH_COSH(x) cosh(x)
#define BI_MATH_TANH(x) tanh(x)
#define BI_MATH_ASINH(x) asin(x)
#define BI_MATH_ACOSH(x) acos(x)
#define BI_MATH_ATANH(x) atan(x)
#else
#define BI_MATH_FABS(x) fabsf(x)
#define BI_MATH_LOG(x) logf(x)
#define BI_MATH_NANLOG(x) (isnan(x) ? logf(0.0f) : logf(x))
#define BI_MATH_EXP(x) expf(x)
#define BI_MATH_NANEXP(x) (isnan(x) ? 0.0f : expf(x))
#define BI_MATH_MAX(x,y) fmaxf(x,y)
#define BI_MATH_MIN(x,y) fminf(x,y)
#define BI_MATH_SQRT(x) sqrtf(x)
#define BI_MATH_POW(x,y) powf(x,y)
#define BI_MATH_FMOD(x,y) fmodf(x,y)
#define BI_MATH_MODF(x,y) modff(x,y)
#define BI_MATH_CEIL(x) ceilf(x)
#define BI_MATH_FLOOR(x) floorf(x)
#define BI_MATH_TGAMMA(x) tgammaf(x)
#define BI_MATH_LGAMMA(x) lgammaf(x)
#define BI_MATH_SIN(x) sinf(x)
#define BI_MATH_COS(x) cosf(x)
#define BI_MATH_TAN(x) tanf(x)
#define BI_MATH_ASIN(x) asinf(x)
#define BI_MATH_ACOS(x) acosf(x)
#define BI_MATH_ATAN(x) atanf(x)
#define BI_MATH_ATAN2(x,y) atan2f(x,y)
#define BI_MATH_SINH(x) sinhf(x)
#define BI_MATH_COSH(x) coshf(x)
#define BI_MATH_TANH(x) tanhf(x)
#define BI_MATH_ASINH(x) asinf(x)
#define BI_MATH_ACOSH(x) acosf(x)
#define BI_MATH_ATANH(x) atanf(x)
#endif

#endif

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
#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

/**
 * @def REAL
 *
 * Cast alias.
 */
#define REAL(x, ...) static_cast<real>(x,##__VA_ARGS__)

/**
 * @def IS_FINITE
 *
 * Is value finite?
 */
#define IS_FINITE(x) ((x)*0 == 0)

#endif

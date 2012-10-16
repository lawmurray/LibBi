/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_SCALAR_HPP
#define BI_MATH_SCALAR_HPP

/**
 * Value type for calculations, float or double. We use this rather
 * than templates because of limited function template support on the GPU
 * and necessary use of global variables.
 */
#ifdef ENABLE_SINGLE
typedef float real;
#else
typedef double real;
#endif

/**
 * @def BI_REAL
 *
 * Cast alias.
 */
#define BI_REAL(x, ...) static_cast<real>(x,##__VA_ARGS__)

#endif

/**
 * @file
 *
 * Miscellaneous math functions.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_MISC_HPP
#define BI_MATH_MISC_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Factorial.
 *
 * @ingroup math
 */
CUDA_FUNC_BOTH int factorial(const int n);

/**
 * Least power of 2 greater than or equal to argument.
 *
 * @param n Argument.
 *
 * @return Least power of 2 greater than or equal to argument.
 */
CUDA_FUNC_BOTH int next_power_2(const int n);

/**
 * Relative error between two floating points.
 */
CUDA_FUNC_BOTH double rel_err(const double a, const double b);

/**
 *
 */
template<class T>
CUDA_FUNC_BOTH bool is_finite(const T x);

}

#include "function.hpp"

inline int bi::factorial(const int n) {
  int result = 1u, y = n;

  while (y > 0) {
    result *= y;
    --y;
  }
  return result;
}

inline int bi::next_power_2(const int n) {
  int result = 1;
  while (result < n) {
    result <<= 1;
  }
  return result;
}

inline double bi::rel_err(const double a, const double b) {
  double diff = bi::abs(a - b);
  double abs_a = bi::abs(a);
  double abs_b = bi::abs(b);

  if (a == b) { // absorbs a == b == 0.0 case
    return 0.0;
  } else {
    return (abs_a > abs_b) ? diff/abs_a : diff/abs_b;
  }
}

template<class T>
inline bool bi::is_finite(const T x) {
  /* having compile and link issues with std::isfinite() or just isfinite()
   * with various compilers, so have rolled own instead */
  T zero = static_cast<T>(0);
  return x*zero == zero;
}

#endif

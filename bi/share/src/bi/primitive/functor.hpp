/**
 * @file
 *
 * Functors for STL and Thrust transformations.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_FUNCTOR_HPP
#define BI_MATH_FUNCTOR_HPP

#include "../math/scalar.hpp"
#include "../math/misc.hpp"
#include "../cuda/cuda.hpp"

#include "thrust/tuple.h"

namespace bi {
/**
 * @ingroup primitive_functor
 *
 * Greater than predicate. NaN are considered less than all values.
 */
template<typename T>
struct nan_greater_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH bool operator()(const T &x, const T& y) const {
    if (isnan(y)) {
      return !isnan(x); // x > y unless isnan(x) too
    } else if (isnan(x)) {
      return false;
    } else {
      return x > y;
    }
  }
};

/**
 * @ingroup primitive_functor
 *
 * Less than predicate. NaN are considered less than all values.
 */
template<typename T>
struct nan_less_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH bool operator()(const T &x, const T& y) const {
    if (isnan(x)) {
      return !isnan(y); // x < y unless isnan(x) too
    } else if (isnan(y)) {
      return false;
    } else {
      return x < y;
    }
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x^2\f$; square unary functor.
 */
template<typename T>
struct square_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return x*x;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\sqrt{x}\f$; square root unary functor.
 */
template<typename T>
struct sqrt_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return BI_MATH_SQRT(x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$1/x\f$; reciprocal unary functor.
 */
template<typename T>
struct rcp_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return BI_MATH_POW(x, -1);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$|x|\f$; absolute value unary functor.
 */
template<typename T>
struct abs_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    //return BI_MATH_FABS(x); // giving "error: calling a host function from a __device__/__global__ function is not allowed" under CUDA 3.2
    return (x < 0) ? -x : x;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x!\f$; factorial unary functor.
 */
template<typename T>
struct factorial_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return factorial(x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\ln (x!)\f$; log factorial unary functor.
 *
 * @param x Integer.
 */
template<typename T>
struct log_factorial_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    T n, result = 0.0;
    for (n = 2; n <= x; ++n) {
      result += BI_MATH_LOG(n);
    }
    return result;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\exp x\f$; exp unary functor.
 */
template<typename T>
struct exp_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return BI_MATH_EXP(x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\exp x\f$; exp unary functor, all nan inputs give zeros as
 * outputs.
 */
template<typename T>
struct nan_exp_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return BI_MATH_NANEXP(x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\ln x\f$; log unary functor.
 */
template<typename T>
struct log_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return BI_MATH_LOG(x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\ln x\f$; log unary functor, all nan inputs give -inf as outputs.
 */
template<typename T>
struct nan_log_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return BI_MATH_NANLOG(x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$kx\f$; constant multiply unary function.
 */
template<typename T>
struct mul_constant_functor : public std::unary_function<T,T> {
  T k;

  mul_constant_functor() {
    //
  }

  mul_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return k*x;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x/k\f$; constant divide unary function.
 */
template<typename T>
struct div_constant_functor : public std::unary_function<T,T> {
  T k;

  div_constant_functor() {
    //
  }

  div_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return x/k;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$y \gets ax + y\f$; element-wise multiply and add.
 */
template<typename T>
struct axpy_functor : public std::unary_function<T,void> {
  T a;

  axpy_functor() {
    //
  }

  axpy_functor(const T a) : a(a) {
    //
  }

  CUDA_FUNC_BOTH void operator()(const T x, const T y) const {
    return a*x + y;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$ax + y\f$; constant multiply and add functor.
 */
template<typename T>
struct axpy_constant_functor : public std::unary_function<T,T> {
  T a, y;

  axpy_constant_functor() {
    //
  }

  axpy_constant_functor(const T a, const T y) : a(a), y(y) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return a*x + y;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$(x - y)/a\f$; constant subtract and divide functor.
 */
template<typename T>
struct invaxpy_constant_functor : public std::unary_function<T,T> {
  T y, a;

  invaxpy_constant_functor() {
    //
  }

  invaxpy_constant_functor(const T y, const T a) : y(y), a(a) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return (x - y)/a;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x \mod k\f$; constant modulus unary function.
 */
template<typename T>
struct modulus_constant_functor : public std::unary_function<T,T> {
  T k;

  modulus_constant_functor() {
    //
  }

  modulus_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return x % k;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x + k\f$; constant add unary function.
 */
template<typename T>
struct add_constant_functor : public std::unary_function<T,T> {
  T k;

  add_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return x + k;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x - k\f$; constant subtract unary function.
 */
template<typename T>
struct sub_constant_functor : public std::unary_function<T,T> {
  T k;

  sub_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return x - k;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x^k\f$; constant power unary function.
 */
template<typename T>
struct pow_constant_functor : public std::unary_function<T,T> {
  T k;

  pow_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return BI_MATH_POW(x, k);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x < k\f$; constant less-than unary predicate.
 */
template<typename T>
struct less_constant_functor : public std::unary_function<T,bool> {
  T k;

  less_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH bool operator()(const T& x) const {
    return x < k;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$x > k\f$; constant greater-than unary predicate.
 */
template<typename T>
struct greater_constant_functor : public std::unary_function<T,bool> {
  T k;

  greater_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH bool operator()(const T& x) const {
    return x > k;
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\exp(2x)\f$; exponentiate and square unary functor. NaN given zero.
 */
template<class T>
struct nan_exp_and_square_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return BI_MATH_NANEXP(BI_REAL(2.0)*x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\exp(x-y)\f$; minus constant and exponentiate unary functor. NaN
 * give zero.
 */
template<class T>
struct nan_minus_and_exp_functor : public std::unary_function<T,T> {
  T y;

  CUDA_FUNC_HOST nan_minus_and_exp_functor(const T y) : y(y) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return BI_MATH_NANEXP(x - y);
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$\exp(2(x-y))\f$; minus constant, exponentiate and square unary
 * functor. NaN give zero.
 */
template<class T>
struct nan_minus_exp_and_square_functor : public std::unary_function<T,T> {
  T y;

  CUDA_FUNC_HOST nan_minus_exp_and_square_functor(const T y) : y(y) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return BI_MATH_NANEXP(static_cast<T>(2.0)*(x - y));
  }
};

/**
 * @ingroup primitive_functor
 *
 * \f$z \exp(x-y)\f$; minus constant, exponentiate and multiply unary functor.
 * NaN give zero.
 */
template<class T>
struct nan_minus_exp_and_multiply_functor : public std::unary_function<T,T> {
  T y, z;

  CUDA_FUNC_HOST nan_minus_exp_and_multiply_functor(const T y, const T z) : y(y),
      z(z) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return z*BI_MATH_NANEXP(x - y);
  }
};

/**
 * @ingroup primitive_functor
 *
 * Is finite predicate.
 */
template<class T>
struct is_finite_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH bool operator()(const T& x) const {
    return is_finite(x);
  }
};

/**
 * @ingroup primitive_functor
 *
 * Is not finite predicate.
 */
template<class T>
struct is_not_finite_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH bool operator()(const T& x) const {
    return !is_finite(x);
  }
};

}

#endif

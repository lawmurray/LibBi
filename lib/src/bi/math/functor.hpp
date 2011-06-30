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

#include "scalar.hpp"
#include "misc.hpp"
#include "../cuda/cuda.hpp"

namespace bi {
/**
 * @ingroup math_functor
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
 * @ingroup math_functor
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
 * @ingroup math_functor
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
 * @ingroup math_functor
 *
 * \f$\sqrt{x}\f$; square root unary functor.
 */
template<typename T>
struct sqrt_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return CUDA_SQRT(x);
  }
};

/**
 * @ingroup math_functor
 *
 * \f$1/x\f$; reciprocal unary functor.
 */
template<typename T>
struct rcp_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return CUDA_POW(x, -1);
  }
};

/**
 * @ingroup math_functor
 *
 * \f$|x|\f$; absolute value unary functor.
 */
template<typename T>
struct abs_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    //return CUDA_ABS(x); // giving "error: calling a host function from a __device__/__global__ function is not allowed" under CUDA 3.2
    return (x < 0) ? -x : x;
  }
};

/**
 * @ingroup math_functor
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
 * @ingroup math_functor
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
      result += CUDA_LOG(n);
    }
    return result;
  }
};

/**
 * @ingroup math_functor
 *
 * \f$\exp x\f$; exp unary functor.
 */
template<typename T>
struct exp_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return CUDA_EXP(x);
  }
};

/**
 * @ingroup math_functor
 *
 * \f$\exp x\f$; exp unary functor, all nan inputs give zeros as
 * outputs.
 */
template<typename T>
struct nan_exp_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return CUDA_NANEXP(x);
  }
};

/**
 * @ingroup math_functor
 *
 * \f$\ln x\f$; log unary functor.
 */
template<typename T>
struct log_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return CUDA_LOG(x);
  }
};

/**
 * @ingroup math_functor
 *
 * \f$\ln x\f$; log unary functor, all nan inputs give -inf as outputs.
 */
template<typename T>
struct nan_log_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T &x) const {
    return CUDA_NANLOG(x);
  }
};

/**
 * @ingroup math_functor
 *
 * \f$kx\f$; constant multiply unary function.
 */
template<typename T>
struct multiply_constant_functor : public std::unary_function<T,T> {
  T k;

  multiply_constant_functor() {
    //
  }

  multiply_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return k*x;
  }
};

/**
 * @ingroup math_functor
 *
 * \f$x/k\f$; constant divide unary function.
 */
template<typename T>
struct divide_constant_functor : public std::unary_function<T,T> {
  T k;

  divide_constant_functor() {
    //
  }

  divide_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return x/k;
  }
};

/**
 * @ingroup math_functor
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
 * @ingroup math_functor
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
 * @ingroup math_functor
 *
 * \f$x - k\f$; constant subtract unary function.
 */
template<typename T>
struct subtract_constant_functor : public std::unary_function<T,T> {
  T k;

  subtract_constant_functor(const T k) : k(k) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return x - k;
  }
};

/**
 * @ingroup math_functor
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
    return CUDA_POW(x, k);
  }
};

/**
 * @ingroup math_functor
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
 * @ingroup math_functor
 *
 * \f$\exp(2x)\f$; exponentiate and square unary functor. NaN given zero.
 */
template<class T>
struct nan_exp_and_square_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH T operator()(const T& x) const {
    return CUDA_NANEXP(REAL(2.0)*x);
  }
};

/**
 * @ingroup math_functor
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
    return CUDA_NANEXP(x - y);
  }
};

/**
 * @ingroup math_functor
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
    return CUDA_NANEXP(static_cast<T>(2.0)*(x - y));
  }
};

/**
 * @ingroup math_functor
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
    return z*CUDA_NANEXP(x - y);
  }
};

/**
 * @ingroup math_functor
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
 * @ingroup math_functor
 *
 * Is not finite predicate.
 */
template<class T>
struct is_not_finite_functor : public std::unary_function<T,T> {
  CUDA_FUNC_BOTH bool operator()(const T& x) const {
    return !is_finite(x);
  }
};

/**
 * @ingroup math_functor
 *
 * Gamma log-density functor.
 */
template<class T>
struct gamma_log_density_functor : public std::unary_function<T,T> {
  const T alpha, beta, logZ;

  CUDA_FUNC_HOST gamma_log_density_functor(const real alpha,
      const real beta, const real logZ) : alpha(alpha), beta(beta),
      logZ(logZ) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    (alpha - 1.0)*CUDA_LOG(x) - x/beta - logZ;
  }
};

/**
 * @ingroup math_functor
 *
 * Inverse gamma log-density functor.
 */
template<class T>
struct inverse_gamma_log_density_functor : public std::unary_function<T,T> {
  const T alpha, beta, logZ;

  CUDA_FUNC_HOST inverse_gamma_log_density_functor(const real alpha,
      const real beta, const real logZ) : alpha(alpha), beta(beta),
      logZ(logZ) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    (-alpha - 1.0)*CUDA_LOG(x) - beta/x - logZ;
  }
};

}

#endif

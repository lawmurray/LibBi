/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2541 $
 * $Date: 2012-05-06 16:19:08 +0800 (Sun, 06 May 2012) $
 */
#ifndef BI_PDF_FUNCTOR_HPP
#define BI_PDF_FUNCTOR_HPP

#include "../math/scalar.hpp"
#include "../math/misc.hpp"

#include "thrust/tuple.h"

namespace bi {

/**
 * @ingroup math_pdf
 *
 * Gamma density functor.
 */
template<class T>
struct gamma_density_functor : public std::unary_function<T,T> {
  const T alpha, beta, logZ;

  CUDA_FUNC_HOST gamma_density_functor(const real alpha,
      const real beta, const real logZ) : alpha(alpha), beta(beta),
      logZ(logZ) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    const T a = 1.0;

    return BI_MATH_EXP((alpha - a)*BI_MATH_LOG(x) - x/beta - logZ);
  }
};

/**
 * @ingroup math_pdf
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
    const T a = 1.0;

    return (alpha - a)*BI_MATH_LOG(x) - x/beta - logZ;
  }
};

/**
 * @ingroup math_pdf
 *
 * Inverse gamma density functor.
 */
template<class T>
struct inverse_gamma_density_functor : public std::unary_function<T,T> {
  const T alpha, beta, logZ;

  CUDA_FUNC_HOST inverse_gamma_density_functor(const real alpha,
      const real beta, const real logZ) : alpha(alpha), beta(beta),
      logZ(logZ) {
    //
  }

  CUDA_FUNC_BOTH T operator()(const T& x) const {
    const T a = 1.0;

    return BI_MATH_EXP((-alpha - a)*BI_MATH_LOG(x) - beta/x - logZ);
  }
};

/**
 * @ingroup math_pdf
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
    const T a = 1.0;

    return (-alpha - a)*BI_MATH_LOG(x) - beta/x - logZ;
  }
};

/**
 * Specialised functor for Gaussian density evaluations.
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
struct gaussian_density_functor {
  /**
   * Constructor.
   *
   * @param logZ Log of normalising constant.
   */
  gaussian_density_functor(const T1 logZ) : logZ(logZ) {
    //
  }

  /**
   * Apply.
   */
  T1 operator()(const T1 p) {
    const T1 a = -0.5;

    return BI_MATH_EXP(a*p - logZ);
  }

  /**
   * Log of normalising constant.
   */
  const T1 logZ;
};

/**
 * Specialised functor for Gaussian density evaluations.
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
struct gaussian_density_update_functor {
  /**
   * Constructor.
   *
   * @param logZ Log of normalising constant.
   */
  gaussian_density_update_functor(const T1 logZ) : logZ(logZ) {
    //
  }

  /**
   * Apply.
   */
  T1 operator()(const T1 p1, const T1 p2) {
    const T1 a = -0.5;

    return p1*BI_MATH_EXP(a*p2 - logZ);
  }

  /**
   * Log of normalising constant.
   */
  const T1 logZ;
};

/**
 * Specialised functor for Gaussian log-density evaluations.
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
struct gaussian_log_density_functor {
  /**
   * Constructor.
   *
   * @param logZ Log of normalising constant.
   */
  gaussian_log_density_functor(const T1 logZ) : logZ(logZ) {
    //
  }

  /**
   * Apply.
   */
  T1 operator()(const T1 p) {
    const T1 a = -0.5;

    return a*p - logZ;
  }

  /**
   * Log of normalising constant.
   */
  const T1 logZ;
};

/**
 * Specialised functor for Gaussian log-density evaluations.
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
struct gaussian_log_density_update_functor {
  /**
   * Constructor.
   *
   * @param logZ Log of normalising constant.
   */
  gaussian_log_density_update_functor(const T1 logZ) : logZ(logZ) {
    //
  }

  /**
   * Apply.
   */
  T1 operator()(const T1 p1, const T1 p2) {
    const T1 a = -0.5;

    return p1 + a*p2 - logZ;
  }

  /**
   * Log of normalising constant.
   */
  const T1 logZ;
};

}

#endif

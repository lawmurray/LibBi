/**
 * @file
 *
 * Math functions for expressions.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_FUNCTION_HPP
#define BI_MATH_FUNCTION_HPP

#include "../cuda/cuda.hpp"

#ifdef __CUDACC__
#include "math_functions.h"
#undef isnan
#else
#include <cmath>
#endif

namespace bi {

CUDA_FUNC_BOTH double abs(const double x);
CUDA_FUNC_BOTH float abs(const float x);
CUDA_FUNC_BOTH int abs(const int x);
CUDA_FUNC_BOTH double log(const double x);
CUDA_FUNC_BOTH float log(const float x);
CUDA_FUNC_BOTH double nanlog(const double x);
CUDA_FUNC_BOTH float nanlog(const float x);
CUDA_FUNC_BOTH double exp(const double x);
CUDA_FUNC_BOTH float exp(const float x);
CUDA_FUNC_BOTH double nanexp(const double x);
CUDA_FUNC_BOTH float nanexp(const float x);
CUDA_FUNC_BOTH double max(const double x, const double y);
CUDA_FUNC_BOTH float max(const float x, const float y);
CUDA_FUNC_BOTH double min(const double x, const double y);
CUDA_FUNC_BOTH float min(const float x, const float y);
CUDA_FUNC_BOTH double sqrt(const double x);
CUDA_FUNC_BOTH float sqrt(const float x);
CUDA_FUNC_BOTH double rsqrt(const double x);
CUDA_FUNC_BOTH float rsqrt(const float x);
CUDA_FUNC_BOTH double pow(const double x, const double y);
CUDA_FUNC_BOTH float pow(const float x, const float y);
CUDA_FUNC_BOTH double mod(const double x, const double y);
CUDA_FUNC_BOTH float mod(const float x, const float y);
CUDA_FUNC_BOTH double ceil(const double x);
CUDA_FUNC_BOTH float ceil(const float x);
CUDA_FUNC_BOTH double floor(const double x);
CUDA_FUNC_BOTH float floor(const float x);
CUDA_FUNC_BOTH double round(const double x);
CUDA_FUNC_BOTH float round(const float x);
CUDA_FUNC_BOTH double gamma(const double x);
CUDA_FUNC_BOTH float gamma(const float x);
CUDA_FUNC_BOTH double lgamma(const double x);
CUDA_FUNC_BOTH float lgamma(const float x);
CUDA_FUNC_BOTH double sin(const double x);
CUDA_FUNC_BOTH float sin(const float x);
CUDA_FUNC_BOTH double cos(const double x);
CUDA_FUNC_BOTH float cos(const float x);
CUDA_FUNC_BOTH double tan(const double x);
CUDA_FUNC_BOTH float tan(const float x);
CUDA_FUNC_BOTH double asin(const double x);
CUDA_FUNC_BOTH float asin(const float x);
CUDA_FUNC_BOTH double acos(const double x);
CUDA_FUNC_BOTH float acos(const float x);
CUDA_FUNC_BOTH double atan(const double x);
CUDA_FUNC_BOTH float atan(const float x);
CUDA_FUNC_BOTH double atan2(const double x, const double y);
CUDA_FUNC_BOTH float atan2(const float x, const double y);
CUDA_FUNC_BOTH double sinh(const double x);
CUDA_FUNC_BOTH float sinh(const float x);
CUDA_FUNC_BOTH double cosh(const double x);
CUDA_FUNC_BOTH float cosh(const float x);
CUDA_FUNC_BOTH double tanh(const double x);
CUDA_FUNC_BOTH float tanh(const float x);
CUDA_FUNC_BOTH double asinh(const double x);
CUDA_FUNC_BOTH float asinh(const float x);
CUDA_FUNC_BOTH double acosh(const double x);
CUDA_FUNC_BOTH float acosh(const float x);
CUDA_FUNC_BOTH double atanh(const double x);
CUDA_FUNC_BOTH float atanh(const float x);
CUDA_FUNC_BOTH double erf(const double x);
CUDA_FUNC_BOTH float erf(const float x);
CUDA_FUNC_BOTH double erfc(const double x);
CUDA_FUNC_BOTH float erfc(const float x);

template<class T>
CUDA_FUNC_BOTH bool isnan(const T x);

template<class T>
CUDA_FUNC_BOTH T max(const T x, const T y);

template<class T>
CUDA_FUNC_BOTH T min(const T x, const T y);

template<class M1, class V1, class V2>
CUDA_FUNC_BOTH void gemv(const M1 A, const V1 x, V2 y);

template<class M1, class M2, class M3>
CUDA_FUNC_BOTH void gemm(const M1 A, const M2 X, M3 Y);

template<class M1, class M2>
CUDA_FUNC_BOTH void trans(const M1 X, M2 Y);

template<class V1, class V2>
CUDA_FUNC_BOTH void inclusive_scan(const V1 x, V2 y);

template<class V1, class V2>
CUDA_FUNC_BOTH void exclusive_scan(const V1 x, V2 y);

}

inline double bi::abs(const double x) {
  return ::fabs(x);
}

inline float bi::abs(const float x) {
  return ::fabsf(x);
}

inline int bi::abs(const int x) {
  return (x >= 0) ? x : -x;
}

inline double bi::log(const double x) {
  return ::log(x);
}

inline float bi::log(const float x) {
  return ::logf(x);
}

inline double bi::nanlog(const double x) {
  return bi::isnan(x) ? bi::log(0.0) : bi::log(x);
}

inline float bi::nanlog(const float x) {
  return bi::isnan(x) ? bi::log(0.0f) : bi::log(x);
}

inline double bi::exp(const double x) {
  return ::exp(x);
}

inline float bi::exp(const float x) {
  return ::expf(x);
}

inline double bi::nanexp(const double x) {
  return bi::isnan(x) ? 0.0 : bi::exp(x);
}

inline float bi::nanexp(const float x) {
  return bi::isnan(x) ? 0.0f : bi::exp(x);
}

inline double bi::max(const double x, const double y) {
  return ::fmax(x, y);
}

inline float bi::max(const float x, const float y) {
  return ::fmaxf(x, y);
}

inline double bi::min(const double x, const double y) {
  return ::fmin(x, y);
}

inline float bi::min(const float x, const float y) {
  return ::fminf(x, y);
}

inline double bi::sqrt(const double x) {
  return ::sqrt(x);
}

inline float bi::sqrt(const float x) {
  return ::sqrtf(x);
}

inline double bi::rsqrt(const double x) {
  #ifdef __CUDA_ARCH__
  return ::rsqrt(x);
  #else
  return bi::pow(x, -0.5);
  #endif
}

inline float bi::rsqrt(const float x) {
  #ifdef __CUDA_ARCH__
  return ::rsqrtf(x);
  #else
  return bi::pow(x, -0.5f);
  #endif
}

inline double bi::pow(const double x, const double y) {
  return ::pow(x, y);
}

inline float bi::pow(const float x, const float y) {
  return ::powf(x, y);
}

inline double bi::mod(const double x, const double y) {
  return ::fmod(x, y);
}

inline float bi::mod(const float x, const float y) {
  return ::fmodf(x, y);
}

inline double bi::ceil(const double x) {
  return ::ceil(x);
}

inline float bi::ceil(const float x) {
  return ::ceilf(x);
}

inline double bi::floor(const double x) {
  return ::floor(x);
}

inline float bi::floor(const float x) {
  return ::floorf(x);
}

inline double bi::round(const double x) {
  return ::round(x);
}

inline float bi::round(const float x) {
  return ::roundf(x);
}

inline double bi::gamma(const double x) {
  return ::tgamma(x);
}

inline float bi::gamma(const float x) {
  return ::tgammaf(x);
}

inline double bi::lgamma(const double x) {
  return ::lgamma(x);
}

inline float bi::lgamma(const float x) {
  return ::lgammaf(x);
}

inline double bi::sin(const double x) {
  return ::sin(x);
}

inline float bi::sin(const float x) {
  return ::sinf(x);
}

inline double bi::cos(const double x) {
  return ::cos(x);
}

inline float bi::cos(const float x) {
  return ::cosf(x);
}

inline double bi::tan(const double x) {
  return ::tan(x);
}

inline float bi::tan(const float x) {
  return ::tanf(x);
}

inline double bi::asin(const double x) {
  return ::asin(x);
}

inline float bi::asin(const float x) {
  return ::asinf(x);
}

inline double bi::acos(const double x) {
  return ::acos(x);
}

inline float bi::acos(const float x) {
  return ::acosf(x);
}

inline double bi::atan(const double x) {
  return ::atan(x);
}

inline float bi::atan(const float x) {
  return ::atanf(x);
}

inline double bi::atan2(const double x, const double y) {
  return ::atan2(x, y);
}

inline float bi::atan2(const float x, const double y) {
  return ::atan2f(x, y);
}

inline double bi::sinh(const double x) {
  return ::sinh(x);
}

inline float bi::sinh(const float x) {
  return ::sinhf(x);
}

inline double bi::cosh(const double x) {
  return ::cosh(x);
}

inline float bi::cosh(const float x) {
  return ::coshf(x);
}

inline double bi::tanh(const double x) {
  return ::tanh(x);
}

inline float bi::tanh(const float x) {
  return ::tanhf(x);
}

inline double bi::asinh(const double x) {
  return ::asinh(x);
}

inline float bi::asinh(const float x) {
  return ::asinhf(x);
}

inline double bi::acosh(const double x) {
  return ::acosh(x);
}

inline float bi::acosh(const float x) {
  return ::acoshf(x);
}

inline double bi::atanh(const double x) {
  return ::atanh(x);
}

inline float bi::atanh(const float x) {
  return ::atanhf(x);
}

inline double bi::erf(const double x){
  return ::erf(x);
}

inline float bi::erf(const float x) {
  return ::erff(x);
}

inline double bi::erfc(const double x){
  return ::erfc(x);
}

inline float bi::erfc(const float x) {
  return ::erfcf(x);
}

template<class T>
inline bool bi::isnan(const T x) {
  // there is no ::isnan(), isnan() is a macro, and std::isnan() is host only
  return x != x;
}

template<class T>
inline T bi::max(const T x, const T y) {
  return (x > y) ? x : y;
}

template<class T>
inline T bi::min(const T x, const T y) {
  return (x < y) ? x : y;
}

template<class M1, class V1, class V2>
inline void bi::gemv(const M1 A, const V1 x, V2 y) {
  /* pre-condition */
  BI_ASSERT(A.size2() == x.size());
  BI_ASSERT(A.size1() == y.size());

  typedef typename V2::value_type T2;

  ///@todo Improve upon naive implementation
  T2 val;
  int i, j;
  for (i = 0; i < A.size1(); ++i) {
    val = static_cast<T2>(0.0);
    for (j = 0; j < A.size2(); ++j) {
      val += A(i,j)*x(j);
    }
    y(i) = val;
  }
}

template<class M1, class M2, class M3>
inline void bi::gemm(const M1 A, const M2 X, M3 Y) {
  /* pre-condition */
  BI_ASSERT(A.size2() == X.size1());
  BI_ASSERT(Y.size1() == A.size1() && Y.size2() == X.size2());

  ///@todo Improve upon naive implementation
  int k;
  for (k = 0; k < X.size2(); ++k) {
    gemv(A, column(X, k), column(Y, k));
  }
}

template<class M1, class M2>
inline void bi::trans(const M1 X, M2 Y) {
  /* pre-condition */
  BI_ASSERT(X.size1() == Y.size2() && X.size2() == Y.size1());

  ///@todo Improve upon naive implementation
  int i, j;
  for (i = 0; i < X.size1(); ++i) {
    for (j = 0; j < X.size2(); ++j) {
      Y(j,i) = X(i,j);
    }
  }
}

template<class V1, class V2>
inline void bi::inclusive_scan(const V1 x, V2 y) {
  /* pre-condition */
  BI_ASSERT(x.size() == y.size());

  typedef typename V2::value_type T2;

  ///@todo Improve numerically upon naive implementation
  int i;
  T2 val = static_cast<T2>(0.0);

  for (i = 0; i < y.size(); ++i) {
    val += x(i);
    y(i) = val;
  }
}

template<class V1, class V2>
inline void bi::exclusive_scan(const V1 x, V2 y) {
  /* pre-condition */
  BI_ASSERT(x.size() == y.size());

  typedef typename V2::value_type T2;

  ///@todo Improve numerically upon naive implementation
  int i;
  T2 val = static_cast<T2>(0.0), next;

  for (i = 0; i < y.size(); ++i) {
    next = x(i); // written this way in case x and y are same vector
    y(i) = val;
    val += next;
  }
}

#endif

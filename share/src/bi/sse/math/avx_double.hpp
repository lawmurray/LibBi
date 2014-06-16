/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_MATH_AVXDOUBLE_HPP
#define BI_SSE_MATH_AVXDOUBLE_HPP

#include "sse_double.hpp"

#include <immintrin.h>

/**
 * @def BI_AVXDOUBLE_UNIVARIATE
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXDOUBLE_UNIVARIATE(func, x) \
    avx_double res; \
    res.unpacked.a = bi::func(x.unpacked.a); \
    res.unpacked.b = bi::func(x.unpacked.b); \
    res.unpacked.c = bi::func(x.unpacked.c); \
    res.unpacked.d = bi::func(x.unpacked.d); \
    return res;

/**
 * @def BI_AVXDOUBLE_BIVARIATE
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXDOUBLE_BIVARIATE(func, x1, x2) \
    avx_double res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2.unpacked.b); \
    res.unpacked.c = bi::func(x1.unpacked.c, x2.unpacked.c); \
    res.unpacked.d = bi::func(x1.unpacked.d, x2.unpacked.d); \
    return res;

/**
 * @def BI_AVXDOUBLE_BIVARIATE_LEFT
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXDOUBLE_BIVARIATE_REAL_RIGHT(func, x1, x2) \
    avx_double res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2); \
    res.unpacked.c = bi::func(x1.unpacked.c, x2); \
    res.unpacked.d = bi::func(x1.unpacked.d, x2); \
    return res;

/**
 * @def BI_AVXDOUBLE_BIVARIATE_REAL_LEFT
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXDOUBLE_BIVARIATE_REAL_LEFT(func, x1, x2) \
    avx_double res; \
    res.unpacked.a = bi::func(x1, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1, x2.unpacked.b); \
    res.unpacked.c = bi::func(x1, x2.unpacked.c); \
    res.unpacked.d = bi::func(x1, x2.unpacked.d); \
    return res;

namespace bi {
/**
 * 128-bit SIMD vector of doubles.
 */
union avx_double {
  struct {
    double a, b, c, d;
  } unpacked;
  __m256d packed;

  avx_double(const __m256d x) :
      packed(x) {
    //
  }

  avx_double(const double a) {
    packed = _mm256_set1_pd(a);
  }

  avx_double() {
    //
  }
};

BI_FORCE_INLINE inline avx_double abs(const avx_double x) {
  return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x.packed);
}

BI_FORCE_INLINE inline avx_double log(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(log, x)
}

BI_FORCE_INLINE inline avx_double nanlog(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(nanlog, x)
}

BI_FORCE_INLINE inline avx_double exp(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(exp, x)
}

BI_FORCE_INLINE inline avx_double nanexp(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(nanexp, x)
}

BI_FORCE_INLINE inline avx_double max(const avx_double x,
    const avx_double y) {
  return _mm256_max_pd(x.packed, y.packed);
}

BI_FORCE_INLINE inline avx_double min(const avx_double x,
    const avx_double y) {
  return _mm256_min_pd(x.packed, y.packed);
}

BI_FORCE_INLINE inline avx_double sqrt(const avx_double x) {
  return _mm256_sqrt_pd(x.packed);
}

BI_FORCE_INLINE inline avx_double pow(const avx_double x,
    const avx_double y) {
  BI_AVXDOUBLE_BIVARIATE(pow, x, y)
}

BI_FORCE_INLINE inline avx_double pow(const avx_double x, const double y) {
  BI_AVXDOUBLE_BIVARIATE_REAL_RIGHT(pow, x, y)
}

BI_FORCE_INLINE inline avx_double pow(const double x, const avx_double y) {
  BI_AVXDOUBLE_BIVARIATE_REAL_LEFT(pow, x, y)
}

BI_FORCE_INLINE inline avx_double mod(const avx_double x,
    const avx_double y) {
  BI_AVXDOUBLE_BIVARIATE(mod, x, y)
}

BI_FORCE_INLINE inline avx_double ceil(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(ceil, x)
}

BI_FORCE_INLINE inline avx_double floor(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(floor, x)
}

BI_FORCE_INLINE inline avx_double gamma(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(gamma, x)
}

BI_FORCE_INLINE inline avx_double lgamma(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(lgamma, x)
}

BI_FORCE_INLINE inline avx_double sin(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(sin, x)
}

BI_FORCE_INLINE inline avx_double cos(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(cos, x)
}

BI_FORCE_INLINE inline avx_double tan(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(tan, x)
}

BI_FORCE_INLINE inline avx_double asin(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(asin, x)
}

BI_FORCE_INLINE inline avx_double acos(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(acos, x)
}

BI_FORCE_INLINE inline avx_double atan(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(atan, x)
}

BI_FORCE_INLINE inline avx_double atan2(const avx_double x,
    const avx_double y) {
  BI_AVXDOUBLE_BIVARIATE(atan2, x, y)
}

BI_FORCE_INLINE inline avx_double sinh(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(sinh, x)
}

BI_FORCE_INLINE inline avx_double cosh(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(cosh, x)
}

BI_FORCE_INLINE inline avx_double tanh(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(tanh, x)
}

BI_FORCE_INLINE inline avx_double asinh(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(asinh, x)
}

BI_FORCE_INLINE inline avx_double acosh(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(acosh, x)
}

BI_FORCE_INLINE inline avx_double atanh(const avx_double x) {
  BI_AVXDOUBLE_UNIVARIATE(atanh, x)
}

BI_FORCE_INLINE inline avx_double& operator+=(avx_double& o1,
    const avx_double& o2) {
  o1.packed = _mm256_add_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_double& operator-=(avx_double& o1,
    const avx_double& o2) {
  o1.packed = _mm256_sub_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_double& operator*=(avx_double& o1,
    const avx_double& o2) {
  o1.packed = _mm256_mul_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_double& operator/=(avx_double& o1,
    const avx_double& o2) {
  o1.packed = _mm256_div_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_double operator+(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_add_pd(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline avx_double operator-(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_sub_pd(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline avx_double operator*(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_mul_pd(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline avx_double operator/(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_div_pd(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline avx_double operator+(const double& o1,
    const avx_double& o2) {
  return _mm256_set1_pd(o1) + o2;
}

BI_FORCE_INLINE inline avx_double operator-(const double& o1,
    const avx_double& o2) {
  return _mm256_set1_pd(o1) - o2;
}

BI_FORCE_INLINE inline avx_double operator*(const double& o1,
    const avx_double& o2) {
  return _mm256_set1_pd(o1) * o2;
}

BI_FORCE_INLINE inline avx_double operator/(const double& o1,
    const avx_double& o2) {
  return _mm256_set1_pd(o1) / o2;
}

BI_FORCE_INLINE inline avx_double operator+(const avx_double& o1,
    const double& o2) {
  return o1 + _mm256_set1_pd(o2);
}

BI_FORCE_INLINE inline avx_double operator-(const avx_double& o1,
    const double& o2) {
  return o1 - _mm256_set1_pd(o2);
}

BI_FORCE_INLINE inline avx_double operator*(const avx_double& o1,
    const double& o2) {
  return o1 * _mm256_set1_pd(o2);
}

BI_FORCE_INLINE inline avx_double operator/(const avx_double& o1,
    const double& o2) {
  return o1 / _mm256_set1_pd(o2);
}

BI_FORCE_INLINE inline avx_double operator==(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_cmp_pd(o1.packed, o2.packed, _CMP_EQ_OQ);
}

BI_FORCE_INLINE inline avx_double operator!=(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_cmp_pd(o1.packed, o2.packed, _CMP_NEQ_OQ);
}

BI_FORCE_INLINE inline avx_double operator<(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_cmp_pd(o1.packed, o2.packed, _CMP_LT_OQ);
}

BI_FORCE_INLINE inline avx_double operator<=(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_cmp_pd(o1.packed, o2.packed, _CMP_LE_OQ);
}

BI_FORCE_INLINE inline avx_double operator>(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_cmp_pd(o1.packed, o2.packed, _CMP_GT_OQ);
}

BI_FORCE_INLINE inline avx_double operator>=(const avx_double& o1,
    const avx_double& o2) {
  return _mm256_cmp_pd(o1.packed, o2.packed, _CMP_GE_OQ);
}

BI_FORCE_INLINE inline const avx_double operator-(const avx_double& o) {
  return _mm256_xor_pd(_mm256_set1_pd(-0.0), o.packed);
}

BI_FORCE_INLINE inline const avx_double operator+(const avx_double& o) {
  return o;
}

}

#endif

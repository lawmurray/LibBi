/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_MATH_SSEDOUBLE_HPP
#define BI_SSE_MATH_SSEDOUBLE_HPP

#include "../../math/scalar.hpp"
#include "../../misc/compile.hpp"

#include <pmmintrin.h>

/**
 * @def BI_SSEDOUBLE_UNIVARIATE
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEDOUBLE_UNIVARIATE(func, x) \
    sse_double res; \
    res.unpacked.a = bi::func(x.unpacked.a); \
    res.unpacked.b = bi::func(x.unpacked.b); \
    return res;

/**
 * @def BI_SSEDOUBLE_BIVARIATE
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEDOUBLE_BIVARIATE(func, x1, x2) \
    sse_double res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2.unpacked.b); \
    return res;

/**
 * @def BI_SSEDOUBLE_BIVARIATE_LEFT
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEDOUBLE_BIVARIATE_REAL_RIGHT(func, x1, x2) \
    sse_double res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2); \
    return res;

/**
 * @def BI_SSEDOUBLE_BIVARIATE_REAL_LEFT
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEDOUBLE_BIVARIATE_REAL_LEFT(func, x1, x2) \
    sse_double res; \
    res.unpacked.a = bi::func(x1, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1, x2.unpacked.b); \
    return res;

namespace bi {
/**
 * 128-bit SIMD vector of doubles.
 */
union sse_double {
  struct {
    double a, b;
  } unpacked;
  __m128d packed;

  sse_double& operator=(const double& o) {
    packed = _mm_set1_pd(o);
    return *this;
  }
};

BI_FORCE_INLINE inline sse_double& operator+=(sse_double& o1,
    const sse_double& o2) {
  o1.packed = _mm_add_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_double& operator-=(sse_double& o1,
    const sse_double& o2) {
  o1.packed = _mm_sub_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_double& operator*=(sse_double& o1,
    const sse_double& o2) {
  o1.packed = _mm_mul_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_double& operator/=(sse_double& o1,
    const sse_double& o2) {
  o1.packed = _mm_div_pd(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_double operator+(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_add_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator-(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_sub_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator*(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_mul_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator/(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_div_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator+(const double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_add_pd(_mm_set1_pd(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator-(const double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_sub_pd(_mm_set1_pd(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator*(const double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_mul_pd(_mm_set1_pd(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator/(const double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_div_pd(_mm_set1_pd(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator+(const sse_double& o1,
    const double& o2) {
  sse_double res;
  res.packed = _mm_add_pd(o1.packed, _mm_set1_pd(o2));
  return res;
}

BI_FORCE_INLINE inline sse_double operator-(const sse_double& o1,
    const double& o2) {
  sse_double res;
  res.packed = _mm_sub_pd(o1.packed, _mm_set1_pd(o2));
  return res;
}

BI_FORCE_INLINE inline sse_double operator*(const sse_double& o1,
    const double& o2) {
  sse_double res;
  res.packed = _mm_mul_pd(o1.packed, _mm_set1_pd(o2));
  return res;
}

BI_FORCE_INLINE inline sse_double operator/(const sse_double& o1,
    const double& o2) {
  sse_double res;
  res.packed = _mm_div_pd(o1.packed, _mm_set1_pd(o2));
  return res;
}

BI_FORCE_INLINE inline sse_double operator==(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_cmpeq_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator!=(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_cmpneq_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator<(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_cmplt_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator<=(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_cmple_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator>(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_cmpgt_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double operator>=(const sse_double& o1,
    const sse_double& o2) {
  sse_double res;
  res.packed = _mm_cmpge_pd(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline const sse_double operator-(const sse_double& o) {
  sse_double res;
  res.packed = _mm_xor_pd(_mm_set1_pd(-0.0), o.packed);
  return res;
}

BI_FORCE_INLINE inline const sse_double operator+(const sse_double& o) {
  return o;
}

BI_FORCE_INLINE inline sse_double abs(const sse_double x) {
  sse_double res;
  res.packed = _mm_andnot_pd(_mm_set1_pd(-0.0), x.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double log(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(log, x)
}

BI_FORCE_INLINE inline sse_double nanlog(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(nanlog, x)
}

BI_FORCE_INLINE inline sse_double exp(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(exp, x)
}

BI_FORCE_INLINE inline sse_double nanexp(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(nanexp, x)
}

BI_FORCE_INLINE inline sse_double max(const sse_double x,
    const sse_double y) {
  sse_double res;
  res.packed = _mm_max_pd(x.packed, y.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double min(const sse_double x,
    const sse_double y) {
  sse_double res;
  res.packed = _mm_min_pd(x.packed, y.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double sqrt(const sse_double x) {
  sse_double res;
  res.packed = _mm_sqrt_pd(x.packed);
  return res;
}

BI_FORCE_INLINE inline sse_double pow(const sse_double x,
    const sse_double y) {
  BI_SSEDOUBLE_BIVARIATE(pow, x, y)
}

BI_FORCE_INLINE inline sse_double pow(const sse_double x, const double y) {
  BI_SSEDOUBLE_BIVARIATE_REAL_RIGHT(pow, x, y)
}

BI_FORCE_INLINE inline sse_double pow(const double x, const sse_double y) {
  BI_SSEDOUBLE_BIVARIATE_REAL_LEFT(pow, x, y)
}

BI_FORCE_INLINE inline sse_double mod(const sse_double x,
    const sse_double y) {
  BI_SSEDOUBLE_BIVARIATE(mod, x, y)
}

BI_FORCE_INLINE inline sse_double ceil(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(ceil, x)
}

BI_FORCE_INLINE inline sse_double floor(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(floor, x)
}

BI_FORCE_INLINE inline sse_double gamma(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(gamma, x)
}

BI_FORCE_INLINE inline sse_double lgamma(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(lgamma, x)
}

BI_FORCE_INLINE inline sse_double sin(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(sin, x)
}

BI_FORCE_INLINE inline sse_double cos(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(cos, x)
}

BI_FORCE_INLINE inline sse_double tan(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(tan, x)
}

BI_FORCE_INLINE inline sse_double asin(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(asin, x)
}

BI_FORCE_INLINE inline sse_double acos(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(acos, x)
}

BI_FORCE_INLINE inline sse_double atan(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(atan, x)
}

BI_FORCE_INLINE inline sse_double atan2(const sse_double x,
    const sse_double y) {
  BI_SSEDOUBLE_BIVARIATE(atan2, x, y)
}

BI_FORCE_INLINE inline sse_double sinh(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(sinh, x)
}

BI_FORCE_INLINE inline sse_double cosh(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(cosh, x)
}

BI_FORCE_INLINE inline sse_double tanh(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(tanh, x)
}

BI_FORCE_INLINE inline sse_double asinh(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(asinh, x)
}

BI_FORCE_INLINE inline sse_double acosh(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(acosh, x)
}

BI_FORCE_INLINE inline sse_double atanh(const sse_double x) {
  BI_SSEDOUBLE_UNIVARIATE(atanh, x)
}

BI_FORCE_INLINE inline double max_reduce(const sse_double x) {
  return bi::max(x.unpacked.a, x.unpacked.b);
}

}

#endif

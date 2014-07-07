/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_MATH_AVXFLOAT_HPP
#define BI_SSE_MATH_AVXFLOAT_HPP

#include "sse_float.hpp"

#include <immintrin.h>

/**
 * @def BI_AVXFLOAT_UNIVARIATE
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXFLOAT_UNIVARIATE(func, x) \
    avx_float res; \
    res.unpacked.a = bi::func(x.unpacked.a); \
    res.unpacked.b = bi::func(x.unpacked.b); \
    return res;

/**
 * @def BI_AVXFLOAT_BIVARIATE
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXFLOAT_BIVARIATE(func, x1, x2) \
    avx_float res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2.unpacked.b); \
    return res;

/**
 * @def BI_AVXFLOAT_BIVARIATE_LEFT
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXFLOAT_BIVARIATE_REAL_RIGHT(func, x1, x2) \
    avx_float res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2); \
    return res;

/**
 * @def BI_AVXFLOAT_BIVARIATE_REAL_LEFT
 *
 * Macro for creating AVX math functions that must operate on individual
 * elements.
 */
#define BI_AVXFLOAT_BIVARIATE_REAL_LEFT(func, x1, x2) \
    avx_float res; \
    res.unpacked.a = bi::func(x1, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1, x2.unpacked.b); \
    return res;

namespace bi {
/**
 * 256-bit SIMD vector of floats.
 */
union avx_float {
  struct {
    sse_float a, b;
  } unpacked;
  __m256 packed;

  avx_float& operator=(const float& o) {
    packed = _mm256_set1_ps(o);
  }
};

BI_FORCE_INLINE inline avx_float& operator+=(avx_float& o1,
    const avx_float& o2) {
  o1.packed = _mm256_add_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_float& operator-=(avx_float& o1,
    const avx_float& o2) {
  o1.packed = _mm256_sub_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_float& operator*=(avx_float& o1,
    const avx_float& o2) {
  o1.packed = _mm256_mul_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_float& operator/=(avx_float& o1,
    const avx_float& o2) {
  o1.packed = _mm256_div_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline avx_float operator+(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_add_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator-(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_sub_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator*(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_mul_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator/(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_div_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator+(const float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_add_ps(_mm256_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator-(const float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_sub_ps(_mm256_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator*(const float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_mul_ps(_mm256_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator/(const float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_div_ps(_mm256_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float operator+(const avx_float& o1,
    const float& o2) {
  avx_float res;
  res.packed = _mm256_add_ps(o1.packed, _mm256_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline avx_float operator-(const avx_float& o1,
    const float& o2) {
  avx_float res;
  res.packed = _mm256_sub_ps(o1.packed, _mm256_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline avx_float operator*(const avx_float& o1,
    const float& o2) {
  avx_float res;
  res.packed = _mm256_mul_ps(o1.packed, _mm256_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline avx_float operator/(const avx_float& o1,
    const float& o2) {
  avx_float res;
  res.packed = _mm256_div_ps(o1.packed, _mm256_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline avx_float operator==(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_cmp_ps(o1.packed, o2.packed, _CMP_EQ_OQ);
  return res;
}

BI_FORCE_INLINE inline avx_float operator!=(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_cmp_ps(o1.packed, o2.packed, _CMP_NEQ_OQ);
  return res;
}

BI_FORCE_INLINE inline avx_float operator<(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_cmp_ps(o1.packed, o2.packed, _CMP_LT_OQ);
  return res;
}

BI_FORCE_INLINE inline avx_float operator<=(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_cmp_ps(o1.packed, o2.packed, _CMP_LE_OQ);
  return res;
}

BI_FORCE_INLINE inline avx_float operator>(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_cmp_ps(o1.packed, o2.packed, _CMP_GT_OQ);
  return res;
}

BI_FORCE_INLINE inline avx_float operator>=(const avx_float& o1,
    const avx_float& o2) {
  avx_float res;
  res.packed = _mm256_cmp_ps(o1.packed, o2.packed, _CMP_GE_OQ);
  return res;
}

BI_FORCE_INLINE inline const avx_float operator-(const avx_float& o) {
  avx_float res;
  res.packed = _mm256_xor_ps(_mm256_set1_ps(-0.0), o.packed);
  return res;
}

BI_FORCE_INLINE inline const avx_float operator+(const avx_float& o) {
  return o;
}

BI_FORCE_INLINE inline avx_float abs(const avx_float x) {
  avx_float res;
  res.packed = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float log(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(log, x)
}

BI_FORCE_INLINE inline avx_float nanlog(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(nanlog, x)
}

BI_FORCE_INLINE inline avx_float exp(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(exp, x)
}

BI_FORCE_INLINE inline avx_float nanexp(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(nanexp, x)
}

BI_FORCE_INLINE inline avx_float max(const avx_float x, const avx_float y) {
  avx_float res;
  res.packed = _mm256_max_ps(x.packed, y.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float min(const avx_float x, const avx_float y) {
  avx_float res;
  res.packed = _mm256_min_ps(x.packed, y.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float sqrt(const avx_float x) {
  avx_float res;
  res.packed = _mm256_sqrt_ps(x.packed);
  return res;
}

BI_FORCE_INLINE inline avx_float pow(const avx_float x, const avx_float y) {
  BI_AVXFLOAT_BIVARIATE(pow, x, y)
}

BI_FORCE_INLINE inline avx_float pow(const avx_float x, const float y) {
  BI_AVXFLOAT_BIVARIATE_REAL_RIGHT(pow, x, y)
}

BI_FORCE_INLINE inline avx_float pow(const float x, const avx_float y) {
  BI_AVXFLOAT_BIVARIATE_REAL_LEFT(pow, x, y)
}

BI_FORCE_INLINE inline avx_float mod(const avx_float x, const avx_float y) {
  BI_AVXFLOAT_BIVARIATE(mod, x, y)
}

BI_FORCE_INLINE inline avx_float ceil(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(ceil, x)
}

BI_FORCE_INLINE inline avx_float floor(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(floor, x)
}

BI_FORCE_INLINE inline avx_float gamma(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(gamma, x)
}

BI_FORCE_INLINE inline avx_float lgamma(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(lgamma, x)
}

BI_FORCE_INLINE inline avx_float sin(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(sin, x)
}

BI_FORCE_INLINE inline avx_float cos(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(cos, x)
}

BI_FORCE_INLINE inline avx_float tan(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(tan, x)
}

BI_FORCE_INLINE inline avx_float asin(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(asin, x)
}

BI_FORCE_INLINE inline avx_float acos(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(acos, x)
}

BI_FORCE_INLINE inline avx_float atan(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(atan, x)
}

BI_FORCE_INLINE inline avx_float atan2(const avx_float x, const avx_float y) {
  BI_AVXFLOAT_BIVARIATE(atan2, x, y)
}

BI_FORCE_INLINE inline avx_float sinh(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(sinh, x)
}

BI_FORCE_INLINE inline avx_float cosh(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(cosh, x)
}

BI_FORCE_INLINE inline avx_float tanh(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(tanh, x)
}

BI_FORCE_INLINE inline avx_float asinh(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(asinh, x)
}

BI_FORCE_INLINE inline avx_float acosh(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(acosh, x)
}

BI_FORCE_INLINE inline avx_float atanh(const avx_float x) {
  BI_AVXFLOAT_UNIVARIATE(atanh, x)
}

BI_FORCE_INLINE inline float max_reduce(const avx_float x) {
  return bi::max(bi::max_reduce(x.unpacked.a), bi::max_reduce(x.unpacked.b));
}

}

#endif

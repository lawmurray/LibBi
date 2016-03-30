/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_MATH_SSEFLOAT_HPP
#define BI_SSE_MATH_SSEFLOAT_HPP

#include "../../math/scalar.hpp"
#include "../../misc/compile.hpp"

#include <pmmintrin.h>

/**
 * @def BI_SSEFLOAT_UNIVARIATE
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEFLOAT_UNIVARIATE(func, x) \
    sse_float res; \
    res.unpacked.a = bi::func(x.unpacked.a); \
    res.unpacked.b = bi::func(x.unpacked.b); \
    res.unpacked.c = bi::func(x.unpacked.c); \
    res.unpacked.d = bi::func(x.unpacked.d); \
    return res;

/**
 * @def BI_SSEFLOAT_BIVARIATE
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEFLOAT_BIVARIATE(func, x1, x2) \
    sse_float res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2.unpacked.b); \
    res.unpacked.c = bi::func(x1.unpacked.c, x2.unpacked.c); \
    res.unpacked.d = bi::func(x1.unpacked.d, x2.unpacked.d); \
    return res;

/**
 * @def BI_SSEFLOAT_BIVARIATE_LEFT
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEFLOAT_BIVARIATE_REAL_RIGHT(func, x1, x2) \
    sse_float res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2); \
    res.unpacked.c = bi::func(x1.unpacked.c, x2); \
    res.unpacked.d = bi::func(x1.unpacked.d, x2); \
    return res;

/**
 * @def BI_SSEFLOAT_BIVARIATE_REAL_LEFT
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#define BI_SSEFLOAT_BIVARIATE_REAL_LEFT(func, x1, x2) \
    sse_float res; \
    res.unpacked.a = bi::func(x1, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1, x2.unpacked.b); \
    res.unpacked.c = bi::func(x1, x2.unpacked.c); \
    res.unpacked.d = bi::func(x1, x2.unpacked.d); \
    return res;

namespace bi {
/**
 * 128-bit SIMD vector of floats.
 */
union sse_float {
  struct {
    float a, b, c, d;
  } unpacked;
  __m128 packed;

  sse_float& operator=(const float& o) {
    packed = _mm_set1_ps(o);
    return *this;
  }
};

BI_FORCE_INLINE inline sse_float& operator+=(sse_float& o1,
    const sse_float& o2) {
  o1.packed = _mm_add_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_float& operator-=(sse_float& o1,
    const sse_float& o2) {
  o1.packed = _mm_sub_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_float& operator*=(sse_float& o1,
    const sse_float& o2) {
  o1.packed = _mm_mul_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_float& operator/=(sse_float& o1,
    const sse_float& o2) {
  o1.packed = _mm_div_ps(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline sse_float operator+(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_add_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator-(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_sub_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator*(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_mul_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator/(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_div_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator+(const float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_add_ps(_mm_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator-(const float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_add_ps(_mm_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator*(const float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_mul_ps(_mm_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator/(const float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_div_ps(_mm_set1_ps(o1), o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator+(const sse_float& o1,
    const float& o2) {
  sse_float res;
  res.packed = _mm_add_ps(o1.packed, _mm_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline sse_float operator-(const sse_float& o1,
    const float& o2) {
  sse_float res;
  res.packed = _mm_sub_ps(o1.packed, _mm_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline sse_float operator*(const sse_float& o1,
    const float& o2) {
  sse_float res;
  res.packed = _mm_mul_ps(o1.packed, _mm_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline sse_float operator/(const sse_float& o1,
    const float& o2) {
  sse_float res;
  res.packed = _mm_div_ps(o1.packed, _mm_set1_ps(o2));
  return res;
}

BI_FORCE_INLINE inline sse_float operator==(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_cmpeq_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator!=(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_cmpneq_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator<(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_cmplt_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator<=(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_cmple_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator>(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_cmpgt_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float operator>=(const sse_float& o1,
    const sse_float& o2) {
  sse_float res;
  res.packed = _mm_cmpge_ps(o1.packed, o2.packed);
  return res;
}

BI_FORCE_INLINE inline const sse_float operator-(const sse_float& o) {
  sse_float res;
  res.packed = _mm_xor_ps(_mm_set1_ps(-0.0f), o.packed);
  return res;
}

BI_FORCE_INLINE inline const sse_float operator+(const sse_float& o) {
  return o;
}

BI_FORCE_INLINE inline sse_float abs(const sse_float x) {
  sse_float res;
  res.packed = _mm_andnot_ps(_mm_set1_ps(-0.0f), x.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float log(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(log, x)
}

BI_FORCE_INLINE inline sse_float nanlog(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(nanlog, x)
}

BI_FORCE_INLINE inline sse_float exp(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(exp, x)
}

BI_FORCE_INLINE inline sse_float nanexp(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(nanexp, x)
}

BI_FORCE_INLINE inline sse_float max(const sse_float x, const sse_float y) {
  sse_float res;
  res.packed = _mm_max_ps(x.packed, y.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float min(const sse_float x, const sse_float y) {
  sse_float res;
  res.packed = _mm_min_ps(x.packed, y.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float sqrt(const sse_float x) {
  sse_float res;
  res.packed = _mm_sqrt_ps(x.packed);
  return res;
}

BI_FORCE_INLINE inline sse_float pow(const sse_float x, const sse_float y) {
  BI_SSEFLOAT_BIVARIATE(pow, x, y)
}

BI_FORCE_INLINE inline sse_float pow(const sse_float x, const float y) {
  BI_SSEFLOAT_BIVARIATE_REAL_RIGHT(pow, x, y)
}

BI_FORCE_INLINE inline sse_float pow(const float x, const sse_float y) {
  BI_SSEFLOAT_BIVARIATE_REAL_LEFT(pow, x, y)
}

BI_FORCE_INLINE inline sse_float mod(const sse_float x, const sse_float y) {
  BI_SSEFLOAT_BIVARIATE(mod, x, y)
}

BI_FORCE_INLINE inline sse_float ceil(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(ceil, x)
}

BI_FORCE_INLINE inline sse_float floor(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(floor, x)
}

BI_FORCE_INLINE inline sse_float gamma(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(gamma, x)
}

BI_FORCE_INLINE inline sse_float lgamma(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(lgamma, x)
}

BI_FORCE_INLINE inline sse_float sin(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(sin, x)
}

BI_FORCE_INLINE inline sse_float cos(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(cos, x)
}

BI_FORCE_INLINE inline sse_float tan(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(tan, x)
}

BI_FORCE_INLINE inline sse_float asin(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(asin, x)
}

BI_FORCE_INLINE inline sse_float acos(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(acos, x)
}

BI_FORCE_INLINE inline sse_float atan(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(atan, x)
}

BI_FORCE_INLINE inline sse_float atan2(const sse_float x, const sse_float y) {
  BI_SSEFLOAT_BIVARIATE(atan2, x, y)
}

BI_FORCE_INLINE inline sse_float sinh(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(sinh, x)
}

BI_FORCE_INLINE inline sse_float cosh(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(cosh, x)
}

BI_FORCE_INLINE inline sse_float tanh(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(tanh, x)
}

BI_FORCE_INLINE inline sse_float asinh(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(asinh, x)
}

BI_FORCE_INLINE inline sse_float acosh(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(acosh, x)
}

BI_FORCE_INLINE inline sse_float atanh(const sse_float x) {
  BI_SSEFLOAT_UNIVARIATE(atanh, x)
}

BI_FORCE_INLINE inline float max_reduce(const sse_float x) {
  return bi::max(bi::max(x.unpacked.a, x.unpacked.b), bi::max(x.unpacked.c, x.unpacked.d));
}

}

#endif

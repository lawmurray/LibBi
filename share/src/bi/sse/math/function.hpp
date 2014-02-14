/**
 * @file
 *
 * Functions for Streaming SIMD Extensions (SSE).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_MATH_FUNCTION_HPP
#define BI_SSE_MATH_FUNCTION_HPP

#include "scalar.hpp"

/**
 * @def BI_SSE_UNIVARIATE
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#ifdef ENABLE_SINGLE
#define BI_SSE_UNIVARIATE(func, x) \
    sse_real res; \
    res.unpacked.a = bi::func(x.unpacked.a); \
    res.unpacked.b = bi::func(x.unpacked.b); \
    res.unpacked.c = bi::func(x.unpacked.c); \
    res.unpacked.d = bi::func(x.unpacked.d); \
    return res;
#else
#define BI_SSE_UNIVARIATE(func, x) \
    sse_real res; \
    res.unpacked.a = bi::func(x.unpacked.a); \
    res.unpacked.b = bi::func(x.unpacked.b); \
    return res;
#endif

/**
 * @def BI_SSE_BIVARIATE
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#ifdef ENABLE_SINGLE
#define BI_SSE_BIVARIATE(func, x1, x2) \
    sse_real res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2.unpacked.b); \
    res.unpacked.c = bi::func(x1.unpacked.c, x2.unpacked.c); \
    res.unpacked.d = bi::func(x1.unpacked.d, x2.unpacked.d); \
    return res;
#else
#define BI_SSE_BIVARIATE(func, x1, x2) \
    sse_real res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2.unpacked.b); \
    return res;
#endif

/**
 * @def BI_SSE_BIVARIATE_LEFT
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#ifdef ENABLE_SINGLE
#define BI_SSE_BIVARIATE_REAL_RIGHT(func, x1, x2) \
    sse_real res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2); \
    res.unpacked.c = bi::func(x1.unpacked.c, x2); \
    res.unpacked.d = bi::func(x1.unpacked.d, x2); \
    return res;
#else
#define BI_SSE_BIVARIATE_REAL_RIGHT(func, x1, x2) \
    sse_real res; \
    res.unpacked.a = bi::func(x1.unpacked.a, x2); \
    res.unpacked.b = bi::func(x1.unpacked.b, x2); \
    return res;
#endif

/**
 * @def BI_SSE_BIVARIATE_REAL_LEFT
 *
 * Macro for creating SSE math functions that must operate on individual
 * elements.
 */
#ifdef ENABLE_SINGLE
#define BI_SSE_BIVARIATE_REAL_LEFT(func, x1, x2) \
    sse_real res; \
    res.unpacked.a = bi::func(x1, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1, x2.unpacked.b); \
    res.unpacked.c = bi::func(x1, x2.unpacked.c); \
    res.unpacked.d = bi::func(x1, x2.unpacked.d); \
    return res;
#else
#define BI_SSE_BIVARIATE_REAL_LEFT(func, x1, x2) \
    sse_real res; \
    res.unpacked.a = bi::func(x1, x2.unpacked.a); \
    res.unpacked.b = bi::func(x1, x2.unpacked.b); \
    return res;
#endif

namespace bi {

//double abs(const double x);
sse_real abs(const bi::sse_real x);
//double log(const double x);
sse_real log(const bi::sse_real x);
//double nanlog(const double x);
sse_real nanlog(const bi::sse_real x);
//double exp(const double x);
sse_real exp(const bi::sse_real x);
//double nanexp(const double x);
sse_real nanexp(const bi::sse_real x);
//double max(const double x, const double y);
sse_real max(const bi::sse_real x, const bi::sse_real y);
//double min(const double x, const double y);
sse_real min(const bi::sse_real x, const bi::sse_real y);
//double sqrt(const double x);
sse_real sqrt(const bi::sse_real x);
//double pow(const double x, const double y);
sse_real pow(const bi::sse_real x, const bi::sse_real y);
sse_real pow(const bi::sse_real x, const real y);
sse_real pow(const real x, const bi::sse_real y);
//double mod(const double x, const double y);
sse_real mod(const bi::sse_real x, const bi::sse_real y);
//double ceil(const double x);
sse_real ceil(const bi::sse_real x);
//double floor(const double x);
sse_real floor(const bi::sse_real x);
//double gamma(const double x);
sse_real gamma(const bi::sse_real x);
//double lgamma(const double x);
sse_real lgamma(const bi::sse_real x);
//double sin(const double x);
sse_real sin(const bi::sse_real x);
//double cos(const double x);
sse_real cos(const bi::sse_real x);
//double tan(const double x);
sse_real tan(const bi::sse_real x);
//double asin(const double x);
sse_real asin(const bi::sse_real x);
//double acos(const double x);
sse_real acos(const bi::sse_real x);
//double atan(const double x);
sse_real atan(const bi::sse_real x);
//double atan2(const double x, const double y);
sse_real atan2(const bi::sse_real x, const bi::sse_real y);
//double sinh(const double x);
sse_real sinh(const bi::sse_real x);
//double cosh(const double x);
sse_real cosh(const bi::sse_real x);
//double tanh(const double x);
sse_real tanh(const bi::sse_real x);
//double asinh(const double x);
sse_real asinh(const bi::sse_real x);
//double acosh(const double x);
sse_real acosh(const bi::sse_real x);
//double atanh(const double x);
sse_real atanh(const bi::sse_real x);
//double heaviside(const double x);
sse_real heaviside(const bi::sse_real x);

}

//inline double bi::abs(const double x) {
//  return ::fabs(x);
//}

inline bi::sse_real bi::abs(const bi::sse_real x) {
  return BI_SSE_ANDNOT_P(BI_SSE_SET1_P(BI_REAL(-0.0)), x.packed);
}

//inline double bi::log(const double x) {
//  return ::log(x);
//}

inline bi::sse_real bi::log(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(log, x)
}

//inline double bi::nanlog(const double x) {
//  return isnan(x) ? log(0.0) : log(x);
//}

inline bi::sse_real bi::nanlog(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(nanlog, x)
}

//inline double bi::exp(const double x) {
//  return ::exp(x);
//}

inline bi::sse_real bi::exp(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(exp, x)
}

//inline double bi::nanexp(const double x) {
//  return isnan(x) ? exp(0.0) : exp(x);
//}

inline bi::sse_real bi::nanexp(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(nanexp, x)
}

//inline double bi::max(const double x, const double y) {
//  return ::fmax(x, y);
//}

inline bi::sse_real bi::max(const bi::sse_real x, const bi::sse_real y) {
  return BI_SSE_MAX_P(x.packed, y.packed);
}

//inline double bi::min(const double x, const double y) {
//  return ::fmin(x, y);
//}

inline bi::sse_real bi::min(const bi::sse_real x, const bi::sse_real y) {
  return BI_SSE_MIN_P(x.packed, y.packed);
}

//inline double bi::sqrt(const double x) {
//  return ::sqrt(x);
//}

inline bi::sse_real bi::sqrt(const bi::sse_real x) {
  return BI_SSE_SQRT_P(x.packed);
}

//inline double bi::pow(const double x, const double y) {
//  return ::pow(x, y);
//}

inline bi::sse_real bi::pow(const bi::sse_real x, const bi::sse_real y) {
  BI_SSE_BIVARIATE(pow, x, y)
}

inline bi::sse_real bi::pow(const bi::sse_real x, const real y) {
  BI_SSE_BIVARIATE_REAL_RIGHT(pow, x, y)
}

inline bi::sse_real bi::pow(const real x, const bi::sse_real y) {
  BI_SSE_BIVARIATE_REAL_LEFT(pow, x, y)
}

//inline double bi::mod(const double x, const double y) {
//  return ::fmod(x, y);
//}

inline bi::sse_real bi::mod(const bi::sse_real x, const bi::sse_real y) {
  BI_SSE_BIVARIATE(mod, x, y)
}

//inline double bi::ceil(const double x) {
//  return ::ceil(x);
//}

inline bi::sse_real bi::ceil(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(ceil, x)
}

//inline double bi::floor(const double x) {
//  return ::floor(x);
//}

inline bi::sse_real bi::floor(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(floor, x)
}

//inline double bi::gamma(const double x) {
//  return ::tgamma(x);
//}

inline bi::sse_real bi::gamma(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(gamma, x)
}

//inline double bi::lgamma(const double x) {
//  return ::lgamma(x);
//}

inline bi::sse_real bi::lgamma(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(lgamma, x)
}

//inline double bi::sin(const double x) {
//  return ::sin(x);
//}

inline bi::sse_real bi::sin(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(sin, x)
}

//inline double bi::cos(const double x) {
//  return ::cos(x);
//}

inline bi::sse_real bi::cos(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(cos, x)
}

//inline double bi::tan(const double x) {
//  return ::tan(x);
//}

inline bi::sse_real bi::tan(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(tan, x)
}

//inline double bi::asin(const double x) {
//  return ::asin(x);
//}

inline bi::sse_real bi::asin(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(asin, x)
}

//inline double bi::acos(const double x) {
//  return ::acos(x);
//}

inline bi::sse_real bi::acos(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(acos, x)
}

//inline double bi::atan(const double x) {
//  return ::atan(x);
//}

inline bi::sse_real bi::atan(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(atan, x)
}

//inline double bi::atan2(const double x, const double y) {
//  return ::atan2(x, y);
//}

inline bi::sse_real bi::atan2(const bi::sse_real x, const bi::sse_real y) {
  BI_SSE_BIVARIATE(atan2, x, y)
}

//inline double bi::sinh(const double x) {
//  return ::sinh(x);
//}

inline bi::sse_real bi::sinh(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(sinh, x)
}

//inline double bi::cosh(const double x) {
//  return ::cosh(x);
//}

inline bi::sse_real bi::cosh(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(cosh, x)
}

//inline double tanh(const double x) {
//  return ::tanh(x);
//}

inline bi::sse_real bi::tanh(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(tanh, x)
}

//inline double asinh(const double x) {
//  return ::asinh(x);
//}

inline bi::sse_real bi::asinh(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(asinh, x)
}

//inline double bi::acosh(const double x) {
//  return ::acosh(x);
//}

inline bi::sse_real bi::acosh(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(acosh, x)
}

//inline double atanh(const double x) {
//  return ::atanh(x);
//}

inline bi::sse_real bi::atanh(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(atanh, x)
}

//inline double heaviside(const double x) {
//  return (x < 0) ? 0 : 1;
//}

inline bi::sse_real bi::heaviside(const bi::sse_real x) {
  BI_SSE_UNIVARIATE(heaviside, x)
}

#endif

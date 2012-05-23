/**
 * @file
 *
 * Types and functions for Streaming SIMD Extensions (SSE). Most of these
 * are forcibly inlined, as they appear to not be otherwise under GCC or
 * Intel.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_SSE_HPP
#define BI_SSE_SSE_HPP

#include "../math/scalar.hpp"
#include "../misc/compile.hpp"

#include <pmmintrin.h>

/**
 * @def BI_SSE_SIZE
 *
 * Number of packed elements in an sse_real variable.
 */
#ifdef ENABLE_DOUBLE
#define BI_SSE_SIZE 2
#else
#define BI_SSE_SIZE 4
#endif

/*
 * Function aliases.
 */
#ifdef ENABLE_DOUBLE
#define BI_SSE_ADD_S _mm_add_sd
#define BI_SSE_ADD_P _mm_add_pd
#define BI_SSE_SUB_S _mm_sub_sd
#define BI_SSE_SUB_P _mm_sub_pd
#define BI_SSE_MUL_S _mm_mul_sd
#define BI_SSE_MUL_P _mm_mul_pd
#define BI_SSE_DIV_S _mm_div_sd
#define BI_SSE_DIV_P _mm_div_pd
#define BI_SSE_SQRT_S _mm_sqrt_sd
#define BI_SSE_SQRT_P _mm_sqrt_pd
#define BI_SSE_ABS_S _mm_abs_sd
#define BI_SSE_ABS_P _mm_abs_pd
#define BI_SSE_MAX_P _mm_max_pd
#define BI_SSE_MIN_P _mm_min_pd
#define BI_SSE_LOAD_S _mm_load_sd
#define BI_SSE_LOAD_P _mm_load_pd
#define BI_SSE_LOAD1_P _mm_load1_pd
#define BI_SSE_STORE_S _mm_store_sd
#define BI_SSE_STORE_P _mm_store_pd
#define BI_SSE_STORE1_P _mm_store1_pd
#define BI_SSE_SET_S _mm_set_sd
#define BI_SSE_SET_P _mm_set_pd
#define BI_SSE_SET1_P _mm_set1_pd
#define BI_SSE_SHUFFLE_P _mm_shuffle_pd
#define BI_SSE_CVT_S _mm_cvtsd_f64 // get first component
#define BI_SSE_CMPEQ_S _mm_cmpeq_sd
#define BI_SSE_CMPEQ_P _mm_cmpeq_pd
#define BI_SSE_CMPLT_S _mm_cmplt_sd
#define BI_SSE_CMPLT_P _mm_cmplt_pd
#define BI_SSE_CMPLE_S _mm_cmple_sd
#define BI_SSE_CMPLE_P _mm_cmple_pd
#define BI_SSE_CMPGT_S _mm_cmpgt_sd
#define BI_SSE_CMPGT_P _mm_cmpgt_pd
#define BI_SSE_CMPGE_S _mm_cmpge_sd
#define BI_SSE_CMPGE_P _mm_cmpge_pd
#define BI_SSE_CMPNEQ_S _mm_cmpneq_sd
#define BI_SSE_CMPNEQ_P _mm_cmpneq_pd
#define BI_SSE_AND_P _mm_and_pd
#define BI_SSE_ANDNOT_P _mm_andnot_pd
#define BI_SSE_HADD_P _mm_hadd_pd // horizontal add
#define BI_SSE_ROTATE_LEFT(x) BI_SSE_SHUFFLE_P(x, x, _MM_SHUFFLE(1,0,3,2))
#else
#define BI_SSE_ADD_S _mm_add_ss
#define BI_SSE_ADD_P _mm_add_ps
#define BI_SSE_SUB_S _mm_sub_ss
#define BI_SSE_SUB_P _mm_sub_ps
#define BI_SSE_MUL_S _mm_mul_ss
#define BI_SSE_MUL_P _mm_mul_ps
#define BI_SSE_DIV_S _mm_div_ss
#define BI_SSE_DIV_P _mm_div_ps
#define BI_SSE_SQRT_S _mm_sqrt_ss
#define BI_SSE_SQRT_P _mm_sqrt_ps
#define BI_SSE_MAX_P _mm_max_ps
#define BI_SSE_MIN_P _mm_min_ps
#define BI_SSE_LOAD_S _mm_load_ss
#define BI_SSE_LOAD_P _mm_load_ps
#define BI_SSE_LOAD1_P _mm_load1_ps
#define BI_SSE_STORE_S _mm_store_ss
#define BI_SSE_STORE_P _mm_store_ps
#define BI_SSE_STORE1_P _mm_store1_ps
#define BI_SSE_SET_S _mm_set_ss
#define BI_SSE_SET_P _mm_set_ps
#define BI_SSE_SET1_P _mm_set1_ps
#define BI_SSE_SHUFFLE_P _mm_shuffle_ps
#define BI_SSE_CVT_S _mm_cvtss_f32 // get first component
#define BI_SSE_CMPEQ_S _mm_cmpeq_ss
#define BI_SSE_CMPEQ_P _mm_cmpeq_ps
#define BI_SSE_CMPLT_S _mm_cmplt_ss
#define BI_SSE_CMPLT_P _mm_cmplt_ps
#define BI_SSE_CMPLE_S _mm_cmple_ss
#define BI_SSE_CMPLE_P _mm_cmple_ps
#define BI_SSE_CMPGT_S _mm_cmpgt_ss
#define BI_SSE_CMPGT_P _mm_cmpgt_ps
#define BI_SSE_CMPGE_S _mm_cmpge_ss
#define BI_SSE_CMPGE_P _mm_cmpge_ps
#define BI_SSE_CMPNEQ_S _mm_cmpneq_ss
#define BI_SSE_CMPNEQ_P _mm_cmpneq_ps
#define BI_SSE_AND_P _mm_and_ps
#define BI_SSE_ANDNOT_P _mm_andnot_ps
#define BI_SSE_HADD_P _mm_hadd_ps
#define BI_SSE_ROTATE_LEFT(x) BI_SSE_SHUFFLE_P(x, x, _MM_SHUFFLE(0,3,2,1))
#endif
#define BI_SSE_PREFETCH _mm_prefetch

namespace bi {
/**
 * Wrapper around __m128 or __m128d 128-bit packed floating point type.
 */
struct sse_real {
  /**
   * 128-bit packed floating point type.
   */
  #ifdef ENABLE_DOUBLE
  typedef __m128d m128;
  #else
  typedef __m128 m128;
  #endif

  /**
   * Underlying value.
   */
  m128 value;

  /**
   * Default constructor.
   */
  sse_real();

  /**
   * Constructor.
   *
   * @param value Underlying value.
   */
  sse_real(const m128& value);

  /**
   * Constructor.
   *
   * @param value Array of @c BI_SSE_SIZE values.
   */
  sse_real(const real* value);

  /**
   * Constructor.
   *
   * @param value Value with which to initialise each component.
   */
  sse_real(const real& value);

  /**
   * Assignment operator.
   */
  sse_real& operator=(const sse_real& o);

  /**
   * Assignment operator.
   *
   * @param value Array of @c BI_SSE_SIZE values.
   */
  sse_real& operator=(const real* value);

  /**
   * Assignment operator.
   *
   * @param value Value to which to set each component.
   */
  sse_real& operator=(const real& o);

  /**
   * Addition compound assignment.
   */
  sse_real& operator+=(const sse_real& o);

  /**
   * Subtraction compound assignment.
   */
  sse_real& operator-=(const sse_real& o);

  /**
   * Multiplication compound assignment.
   */
  sse_real& operator*=(const sse_real& o);

  /**
   * Division compound assignment.
   */
  sse_real& operator/=(const sse_real& o);

  /**
   * Addition.
   */
  const sse_real operator+(const sse_real& o) const;

  /**
   * Subtraction.
   */
  const sse_real operator-(const sse_real& o) const;

  /**
   * Multiplication.
   */
  const sse_real operator*(const sse_real& o) const;

  /**
   * Division.
   */
  const sse_real operator/(const sse_real& o) const;

  /**
   * Store values to array.
   */
  void store(real* values) const;

  /**
   * Store single value.
   */
  void store(real& value) const;

};

}

#include "../misc/compile.hpp"
#include "../cuda/cuda.hpp"

BI_FORCE_INLINE inline bi::sse_real::sse_real() {
  //
}

BI_FORCE_INLINE inline bi::sse_real::sse_real(const m128& value) : value(value) {
  //
}

BI_FORCE_INLINE inline bi::sse_real::sse_real(const real* value) :
    value(BI_SSE_LOAD_P(value)) {
  //
}

BI_FORCE_INLINE inline bi::sse_real::sse_real(const real& value) :
    value(BI_SSE_LOAD1_P(&value)) {
  //
}

BI_FORCE_INLINE inline bi::sse_real& bi::sse_real::operator=(const sse_real& o) {
  value = o.value;
  return *this;
}

BI_FORCE_INLINE inline bi::sse_real& bi::sse_real::operator=(const real* value) {
  this->value = BI_SSE_LOAD_P(value);
  return *this;
}

BI_FORCE_INLINE inline bi::sse_real& bi::sse_real::operator=(const real& value) {
  this->value = BI_SSE_LOAD1_P(&value);
  return *this;
}

BI_FORCE_INLINE inline bi::sse_real& bi::sse_real::operator+=(const sse_real& o) {
  value = BI_SSE_ADD_P(value, o.value);
  return *this;
}

BI_FORCE_INLINE inline bi::sse_real& bi::sse_real::operator-=(const sse_real& o) {
  value = BI_SSE_SUB_P(value, o.value);
  return *this;
}

BI_FORCE_INLINE inline bi::sse_real& bi::sse_real::operator*=(const sse_real& o) {
  value = BI_SSE_MUL_P(value, o.value);
  return *this;
}

BI_FORCE_INLINE inline bi::sse_real& bi::sse_real::operator/=(const sse_real& o) {
  value = BI_SSE_DIV_P(value, o.value);
  return *this;
}

BI_FORCE_INLINE inline const bi::sse_real bi::sse_real::operator+(const sse_real& o) const {
  return sse_real(BI_SSE_ADD_P(value, o.value));
}

BI_FORCE_INLINE inline const bi::sse_real bi::sse_real::operator-(const sse_real& o) const {
  return sse_real(BI_SSE_SUB_P(value, o.value));
}

BI_FORCE_INLINE inline const bi::sse_real bi::sse_real::operator*(const sse_real& o) const {
  return sse_real(BI_SSE_MUL_P(value, o.value));
}

BI_FORCE_INLINE inline const bi::sse_real bi::sse_real::operator/(const sse_real& o) const {
  return sse_real(BI_SSE_DIV_P(value, o.value));
}

BI_FORCE_INLINE inline void bi::sse_real::store(real* values) const {
  BI_SSE_STORE_P(values, this->value);
}

BI_FORCE_INLINE inline void bi::sse_real::store(real& value) const {
  BI_SSE_STORE_S(&value, this->value);
}

namespace bi {
/**
 * Square root of packed value.
 */
BI_FORCE_INLINE inline sse_real sse_sqrt(const sse_real& o) {
  return sse_real(BI_SSE_SQRT_P(o.value));
}

/**
 * Pair-wise maximums of packed value.
 */
BI_FORCE_INLINE inline sse_real sse_max(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_MAX_P(o1.value, o2.value));
}

/**
 * Pair-wise minimums of packed value.
 */
BI_FORCE_INLINE inline sse_real sse_min(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_MIN_P(o1.value, o2.value));
}

/**
 * Equality comparison.
 */
BI_FORCE_INLINE inline sse_real sse_eq(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_CMPEQ_P(o1.value, o2.value));
}

/**
 * Non-equality comparison.
 */
BI_FORCE_INLINE inline sse_real sse_neq(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_CMPNEQ_P(o1.value, o2.value));
}

/**
 * Less-than comparison.
 */
BI_FORCE_INLINE inline sse_real sse_lt(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_CMPLT_P(o1.value, o2.value));
}

/**
 * Less-than or equal comparison.
 */
BI_FORCE_INLINE inline sse_real sse_le(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_CMPLE_P(o1.value, o2.value));
}

/**
 * Greater-than comparison.
 */
BI_FORCE_INLINE inline sse_real sse_gt(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_CMPGT_P(o1.value, o2.value));
}

/**
 * Greater-than or equal comparison.
 */
BI_FORCE_INLINE inline sse_real sse_ge(const sse_real& o1, const sse_real& o2) {
  return sse_real(BI_SSE_CMPGE_P(o1.value, o2.value));
}

/**
 * Log of packed value.
 */
BI_FORCE_INLINE inline sse_real sse_log(const sse_real& o) {
  CUDA_ALIGN(16) real x[BI_SSE_SIZE] BI_ALIGN(16);
  o.store(x);
  for (int i = 0; i < BI_SSE_SIZE; ++i) {
    x[i] = BI_MATH_LOG(x[i]);
  }
  return sse_real(x);
}

/**
 * Exp of packed value.
 */
BI_FORCE_INLINE inline sse_real sse_exp(const sse_real& o) {
  CUDA_ALIGN(16) real x[BI_SSE_SIZE] BI_ALIGN(16);
  o.store(x);
  for (int i = 0; i < BI_SSE_SIZE; ++i) {
    x[i] = BI_MATH_EXP(x[i]);
  }
  return sse_real(x);
}

/**
 * Power of packed value to packed value.
 */
BI_FORCE_INLINE inline sse_real sse_pow(const sse_real& o, const sse_real& p) {
  CUDA_ALIGN(16) real x[BI_SSE_SIZE] BI_ALIGN(16);
  CUDA_ALIGN(16) real y[BI_SSE_SIZE] BI_ALIGN(16);
  o.store(x);
  p.store(y);
  for (int i = 0; i < BI_SSE_SIZE; ++i) {
    x[i] = BI_MATH_POW(x[i], y[i]);
  }
  return sse_real(x);
}

/**
 * Power of packed value to scalar.
 */
BI_FORCE_INLINE inline sse_real sse_pow(const sse_real& o, const real& p) {
  CUDA_ALIGN(16) real x[BI_SSE_SIZE] BI_ALIGN(16);
  o.store(x);
  for (int i = 0; i < BI_SSE_SIZE; ++i) {
    x[i] = BI_MATH_POW(x[i], p);
  }
  return sse_real(x);
}

/**
 * Absolute value of packed value.
 *
 * @todo Implement with mask.
 */
BI_FORCE_INLINE inline sse_real sse_fabs(const sse_real& o) {
  sse_real abs_mask(BI_SSE_SET1_P(BI_REAL(0.0)));
  return sse_real(BI_SSE_ANDNOT_P(abs_mask.value, o.value));
}

/**
 * Conditional.
 *
 * @param mask Mask, giving 0x0 for false value, OxF..F for true value.
 * @param o1 Values to assume for true components.
 * @param o2 Values to assume for false components.
 */
BI_FORCE_INLINE inline sse_real sse_if(const sse_real& mask, const sse_real& o1,
    const sse_real& o2) {
  return sse_real(BI_SSE_ADD_P(BI_SSE_AND_P(mask.value, o1.value),
      BI_SSE_ANDNOT_P(mask.value, o2.value)));
}

/**
 * Any.
 *
 * @return True if any mask components are true, false otherwise.
 *
 * @todo Is there a reduction intrinsic for this in SSE3?
 */
BI_FORCE_INLINE inline bool sse_any(const sse_real& mask) {
  bool result = false;
  #ifdef ENABLE_DOUBLE
  CUDA_ALIGN(16) long x[BI_SSE_SIZE] BI_ALIGN(16);
  #else
  CUDA_ALIGN(16) int x[BI_SSE_SIZE] BI_ALIGN(16);
  #endif
  mask.store((real*)x);
  for (int i = 0; i < BI_SSE_SIZE; ++i) {
    result = result || x[i];
  }
  return result;
}

}

/*
 * Overloads for standard math functions.
 */
BI_FORCE_INLINE inline bi::sse_real sqrt(const bi::sse_real& o) {
  return bi::sse_sqrt(o);
}

BI_FORCE_INLINE inline bi::sse_real log(const bi::sse_real& o) {
  return bi::sse_log(o);
}

BI_FORCE_INLINE inline bi::sse_real exp(const bi::sse_real& o) {
  return bi::sse_exp(o);
}

BI_FORCE_INLINE inline bi::sse_real pow(const bi::sse_real& o, const bi::sse_real& p) {
  return bi::sse_pow(o,p);
}

BI_FORCE_INLINE inline bi::sse_real pow(const bi::sse_real& o, const real& p) {
  return bi::sse_pow(o,p);
}

BI_FORCE_INLINE inline bi::sse_real pow(const real& o, const bi::sse_real& p) {
  return bi::sse_pow(bi::sse_real(o),p);
}

BI_FORCE_INLINE inline bi::sse_real fabs(const bi::sse_real& o) {
  return bi::sse_fabs(o);
}

BI_FORCE_INLINE inline bi::sse_real sqrtf(const bi::sse_real& o) {
  return bi::sse_sqrt(o);
}

BI_FORCE_INLINE inline bi::sse_real logf(const bi::sse_real& o) {
  return bi::sse_log(o);
}

BI_FORCE_INLINE inline bi::sse_real expf(const bi::sse_real& o) {
  return bi::sse_exp(o);
}

BI_FORCE_INLINE inline bi::sse_real powf(const bi::sse_real& o, const bi::sse_real& p) {
  return bi::sse_pow(o,p);
}

BI_FORCE_INLINE inline bi::sse_real powf(const bi::sse_real& o, const real& p) {
  return bi::sse_pow(o,p);
}

BI_FORCE_INLINE inline bi::sse_real powf(const real& o, const bi::sse_real& p) {
  return bi::sse_pow(bi::sse_real(o), p);
}

BI_FORCE_INLINE inline bi::sse_real fabsf(const bi::sse_real& o) {
  return bi::sse_fabs(o);
}

/*
 * Overloads for operators when real is on left.
 */
BI_FORCE_INLINE inline const bi::sse_real operator+(const real& x1, const bi::sse_real& x2) {
  bi::sse_real x0(x1);
  return x0 + x2;
}

BI_FORCE_INLINE inline const bi::sse_real operator-(const real& x1, const bi::sse_real& x2) {
  bi::sse_real x0(x1);
  return x0 - x2;
}

BI_FORCE_INLINE inline const bi::sse_real operator*(const real& x1, const bi::sse_real& x2) {
  bi::sse_real x0(x1);
  return x0*x2;
}

BI_FORCE_INLINE inline const bi::sse_real operator/(const real& x1, const bi::sse_real& x2) {
  bi::sse_real x0(x1);
  return x0/x2;
}

/*
 * Overloads for unary operators.
 */
BI_FORCE_INLINE inline const bi::sse_real operator-(const bi::sse_real& x) {
  return x*BI_REAL(-1.0);
}

BI_FORCE_INLINE inline const bi::sse_real operator+(const bi::sse_real& x) {
  return x;
}

#endif

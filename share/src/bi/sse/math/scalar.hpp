/**
 * @file
 *
 * Types and operators for Streaming SIMD Extensions (SSE).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_MATH_SCALAR_HPP
#define BI_SSE_MATH_SCALAR_HPP

#include "../../math/scalar.hpp"
#include "../../misc/compile.hpp"

#include <pmmintrin.h>

/**
 * @def BI_SSE_SIZE
 *
 * Number of packed elements in an sse_real variable.
 */
#ifdef ENABLE_SINGLE
#define BI_SSE_SIZE 4
#else
#define BI_SSE_SIZE 2
#endif

/*
 * Function aliases.
 */
#ifdef ENABLE_SINGLE
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
#define BI_SSE_XOR_P _mm_xor_ps
#define BI_SSE_HADD_P _mm_hadd_ps
#define BI_SSE_ROTATE_LEFT(x) BI_SSE_SHUFFLE_P(x, x, _MM_SHUFFLE(0,3,2,1))
#else
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
#define BI_SSE_XOR_P _mm_xor_pd
#define BI_SSE_HADD_P _mm_hadd_pd // horizontal add
#define BI_SSE_ROTATE_LEFT(x) BI_SSE_SHUFFLE_P(x, x, _MM_SHUFFLE(1,0,3,2))
#endif
#define BI_SSE_PREFETCH _mm_prefetch

namespace bi {
  /**
   * 128-bit packed floating point type.
   */
  union sse_real {
    #ifdef ENABLE_SINGLE
    struct {
      real a, b, c, d;
    } unpacked;
    __m128 packed;

    sse_real(const real a, const real b, const real c, const real d) {
      unpacked.a = a;
      unpacked.b = b;
      unpacked.c = c;
      unpacked.d = d;
    }

    sse_real(const __m128 x) : packed(x) {
      //
    }
    #else
    struct {
      real a, b;
    } unpacked;
    __m128d packed;

    sse_real(const real a, const real b) {
      unpacked.a = a;
      unpacked.b = b;
    }

    sse_real(const __m128d x) : packed(x) {
      //
    }

    real& operator[](const int i) {
      if (i == 0) {
        return unpacked.a;
      } else {
        return unpacked.b;
      }
    }
    #endif

    sse_real() {
      //
    }

    sse_real(const real a) {
      packed = BI_SSE_SET1_P(a);
    }

    sse_real& operator=(const real a) {
      packed = BI_SSE_SET1_P(a);
      return *this;
    }
  };

  sse_real& operator+=(sse_real& o1, const sse_real& o2);
  sse_real& operator-=(sse_real& o1, const sse_real& o2);
  sse_real& operator*=(sse_real& o1, const sse_real& o2);
  sse_real& operator/=(sse_real& o1, const sse_real& o2);
  sse_real operator+(const sse_real& o1, const sse_real& o2);
  sse_real operator-(const sse_real& o1, const sse_real& o2);
  sse_real operator*(const sse_real& o1, const sse_real& o2);
  sse_real operator/(const sse_real& o1, const sse_real& o2);
  sse_real operator+(const real& o1, const sse_real& o2);
  sse_real operator-(const real& o1, const sse_real& o2);
  sse_real operator*(const real& o1, const sse_real& o2);
  sse_real operator/(const real& o1, const sse_real& o2);
  sse_real operator+(const sse_real& o1, const real& o2);
  sse_real operator-(const sse_real& o1, const real& o2);
  sse_real operator*(const sse_real& o1, const real& o2);
  sse_real operator/(const sse_real& o1, const real& o2);
  sse_real operator==(const sse_real& o1, const sse_real& o2);
  sse_real operator!=(const sse_real& o1, const sse_real& o2);
  sse_real operator<(const sse_real& o1, const sse_real& o2);
  sse_real operator<=(const sse_real& o1, const sse_real& o2);
  sse_real operator>(const sse_real& o1, const sse_real& o2);
  sse_real operator>=(const sse_real& o1, const sse_real& o2);
  const sse_real operator-(const sse_real& o);
  const sse_real operator+(const sse_real& o);
}

BI_FORCE_INLINE inline bi::sse_real& bi::operator+=(bi::sse_real& o1, const bi::sse_real& o2) {
  o1.packed = BI_SSE_ADD_P(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline bi::sse_real& bi::operator-=(bi::sse_real& o1, const bi::sse_real& o2) {
  o1.packed = BI_SSE_SUB_P(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline bi::sse_real& bi::operator*=(bi::sse_real& o1, const bi::sse_real& o2) {
  o1.packed = BI_SSE_MUL_P(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline bi::sse_real& bi::operator/=(bi::sse_real& o1, const bi::sse_real& o2) {
  o1.packed = BI_SSE_DIV_P(o1.packed, o2.packed);
  return o1;
}

BI_FORCE_INLINE inline bi::sse_real bi::operator+(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_ADD_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator-(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_SUB_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator*(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_MUL_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator/(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_DIV_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator+(const real& o1, const bi::sse_real& o2) {
  return BI_SSE_SET1_P(o1) + o2;
}

BI_FORCE_INLINE inline bi::sse_real bi::operator-(const real& o1, const bi::sse_real& o2) {
  return BI_SSE_SET1_P(o1) - o2;
}

BI_FORCE_INLINE inline bi::sse_real bi::operator*(const real& o1, const bi::sse_real& o2) {
  return BI_SSE_SET1_P(o1)*o2;
}

BI_FORCE_INLINE inline bi::sse_real bi::operator/(const real& o1, const bi::sse_real& o2) {
  return BI_SSE_SET1_P(o1)/o2;
}

BI_FORCE_INLINE inline bi::sse_real bi::operator+(const bi::sse_real& o1, const real& o2) {
  return o1 + BI_SSE_SET1_P(o2);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator-(const bi::sse_real& o1, const real& o2) {
  return o1 - BI_SSE_SET1_P(o2);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator*(const bi::sse_real& o1, const real& o2) {
  return o1*BI_SSE_SET1_P(o2);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator/(const bi::sse_real& o1, const real& o2) {
  return o1/BI_SSE_SET1_P(o2);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator==(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_CMPEQ_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator!=(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_CMPNEQ_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator<(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_CMPLT_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator<=(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_CMPLE_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator>(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_CMPGT_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline bi::sse_real bi::operator>=(const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_CMPGE_P(o1.packed, o2.packed);
}

BI_FORCE_INLINE inline const bi::sse_real bi::operator-(const bi::sse_real& o) {
  return BI_SSE_XOR_P(BI_SSE_SET1_P(BI_REAL(-0.0)), o.packed);
}

BI_FORCE_INLINE inline const bi::sse_real bi::operator+(const bi::sse_real& o) {
  return o;
}

#endif

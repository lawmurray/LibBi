/**
 * @file
 *
 * Control functions for expressions.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_CONTROL_HPP
#define BI_MATH_CONTROL_HPP

#include "scalar.hpp"

namespace bi {
/**
 * Conditional.
 *
 * @param mask Mask, giving 0x0 for false value, OxF..F for true value.
 * @param o1 Values to assume for true components.
 * @param o2 Values to assume for false components.
 */
sse_real sse_if(const sse_real& mask, const sse_real& o1,
    const sse_real& o2);

/**
 * Any.
 *
 * @return True if any mask components are true, false otherwise.
 *
 * @todo Is there a reduction intrinsic for this in SSE3?
 */
bool sse_any(const sse_real& mask);

}

inline bi::sse_real bi::sse_if(const bi::sse_real& mask,
    const bi::sse_real& o1, const bi::sse_real& o2) {
  return BI_SSE_ADD_P(BI_SSE_AND_P(mask.packed, o1.packed),
      BI_SSE_ANDNOT_P(mask.packed, o2.packed));
}

inline bool bi::sse_any(const bi::sse_real& mask) {
  bool result = false;
  #ifdef ENABLE_SINGLE
  CUDA_ALIGN(16) int x[BI_SSE_SIZE] BI_ALIGN(16);
  #else
  CUDA_ALIGN(16) long x[BI_SSE_SIZE] BI_ALIGN(16);
  #endif
  BI_SSE_STORE_P(reinterpret_cast<real*>(x), mask.packed);
  for (int i = 0; !result && i < BI_SSE_SIZE; ++i) {
    result = result || x[i];
  }
  return result;
}

#endif

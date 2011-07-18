/**
 * @file
 *
 * Miscellaneous functions required by state module.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1634 $
 * $Date: 2011-06-15 11:29:55 +0800 (Wed, 15 Jun 2011) $
 */
#ifndef BI_STATE_MISC_HPP
#define BI_STATE_MISC_HPP

#ifdef USE_SSE
#include "../math/sse.hpp"
#endif

namespace bi {
  /**
   * Round up number of trajectories as required by implementation.
   *
   * @tparam L Location.
   *
   * @param P Minimum number of trajectories.
   *
   * @return Number of trajectories.
   *
   * The following rules are applied:
   *
   * @li for @p L on device, @p P must be either less than 32, or a
   * multiple of 32, and
   * @li for @p L on host with SSE enabled, @p P must be zero, one or a
   * multiple of four (single precision) or two (double precision).
   */
  template<Location L>
  int roundup(const int P) {
    int P1 = P;
    if (L == ON_DEVICE) {
      /* either < 32 or a multiple of 32 number of trajectories required */
      if (P1 > 32) {
        P1 = ((P1 + 31)/32)*32;
      }
    } else {
      #if defined(USE_CPU) and defined(USE_SSE)
      /* zero, one or a multiple of 4 (single precision) or 2 (double
       * precision) required */
      if (P1 > 1) {
        P1 = ((P1 + BI_SSE_SIZE - 1)/BI_SSE_SIZE)*BI_SSE_SIZE;
      }
      #endif
    }

    return P1;
  }
}

#endif

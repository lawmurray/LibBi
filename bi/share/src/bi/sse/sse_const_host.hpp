/**
 * @file
 *
 * Functions for reading of constant state objects through main memory for SSE
 * instructions. Use const_host.hpp methods to bind. Analogue of constant.cuh
 * for device memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_SSECONSTHOST_HPP
#define BI_SSE_SSECONSTHOST_HPP

#include "sse.hpp"
#include "../misc/macro.hpp"

namespace bi {
/**
 * Fetch node values from constant main memory as 128-bit SSE value.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param cox Serial coordinate.
 *
 * @return 2- (double precision) or 4- (single precision) component packed
 * vector of node values beginning at trajectory @p p.
 */
template<class B, class X>
sse_real sse_const_host_fetch(const int cox);

/**
 * Facade for state as 128-bit SSE values in constant main memory.
 *
 * @ingroup state_host
 */
struct sse_const_host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X>
  BI_FORCE_INLINE static sse_real fetch(const int p, const int cox) {
    return sse_const_host_fetch<B,X>(cox);
  }
};

}

#include "sse_state.hpp"

template<class B, class X>
BI_FORCE_INLINE inline bi::sse_real bi::sse_const_host_fetch(
    const int cox) {
  const int i = var_net_start<B,X>::value + cox;
  sse_real result;

  if (is_p_var<X>::value) {
    result = sse_state_get(constHostPState, 0, i);
  } else if (is_px_var<X>::value) {
    result = sse_state_get(constHostPXState, 0, i);
  } else {
    BI_ASSERT(false, "Unsupported node type");
  }
  return result;
}

#endif

/**
 * @file
 *
 * Functions for reading of state objects through main memory for SSE
 * instructions. Use host.hpp methods to bind.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_SSEHOST_HPP
#define BI_SSE_SSEHOST_HPP

#include "sse.hpp"
#include "../misc/macro.hpp"

namespace bi {
/**
 * Fetch node values from main memory as 128-bit SSE value.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p First trajectory id.
 * @param cox Serial coordinate.
 *
 * @return 2- (double precision) or 4- (single precision) component packed
 * vector of node values beginning at trajectory @p p.
 */
template<class B, class X>
sse_real sse_host_fetch(const int p, const int cox);

/**
 * Set node values as 128-bit SSE value in main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p First trajectory id.
 * @param cox Serial coordinate.
 *
 * @param val Value to set for the given node.
 */
template<class B, class X>
void sse_host_put(const int p, const int cox, const sse_real& val);

/**
 * Facade for state as 128-bit SSE values in main memory.
 *
 * @ingroup state_host
 */
struct sse_host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X>
  BI_FORCE_INLINE static sse_real fetch(const int p, const int cox) {
    return sse_host_fetch<B,X>(p, cox);
  }

  /**
   * Put value.
   */
  template<class B, class X>
  BI_FORCE_INLINE static void put(const int p, const int cox,
      const sse_real& val) {
    sse_host_put<B,X>(p, cox, val);
  }
};

}

#include "sse_state.hpp"
#include "../host/host.hpp"
#include "../traits/var_traits.hpp"

template<class B, class X>
BI_FORCE_INLINE inline bi::sse_real bi::sse_host_fetch(const int p,
    const int cox) {
  const int i = var_net_start<B,X>::value + cox;
  sse_real result;

  if (is_d_var<X>::value) {
    result = sse_state_get(hostDState, p, i);
  } else if (is_dx_var<X>::value) {
    result = sse_state_get(hostDXState, p, i);
  } else if (is_r_var<X>::value) {
    result = sse_state_get(hostRState, p, i);
  } else if (is_f_var<X>::value) {
    result = sse_state_get(hostFState, 0, i);
  } else if (is_o_var<X>::value) {
    result = sse_state_get(hostOState, p, i);
  } else if (is_p_var<X>::value) {
    result = sse_state_get(hostPState, 0, i);
  } else if (is_px_var<X>::value) {
    result = sse_state_get(hostPXState, 0, i);
  }
  return result;
}

template<class B, class X>
BI_FORCE_INLINE inline void bi::sse_host_put(const int p,
    const int cox, const sse_real& val) {
  const int i = var_net_start<B,X>::value + cox;

  if (is_d_var<X>::value) {
    sse_state_set(hostDState, p, i, val);
  } else if (is_dx_var<X>::value) {
    sse_state_set(hostDXState, p, i, val);
  } else if (is_r_var<X>::value) {
    sse_state_set(hostRState, p, i, val);
  } else if (is_f_var<X>::value) {
    sse_state_set(hostFState, 0, i, val);
  } else if (is_o_var<X>::value) {
    sse_state_set(hostOState, p, i, val);
  } else if (is_p_var<X>::value) {
    sse_state_set(hostPState, 0, i, val);
  } else if (is_px_var<X>::value) {
    sse_state_set(hostPXState, 0, i, val);
  }
}

#endif

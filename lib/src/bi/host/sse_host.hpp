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
#ifndef BI_HOST_SSE_HOST_HPP
#define BI_HOST_SSE_HOST_HPP

#include "../math/sse.hpp"
#include "../misc/macro.hpp"
#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Fetch node values from main memory as 128-bit SSE value.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @param p First trajectory id.
 * @param cox Base coordinates.
 *
 * @return 2- (double precision) or 4- (single precision) component packed
 * vector of node values beginning at trajectory @p p.
 */
template<class B, class X, int Xo, int Yo, int Zo>
sse_real sse_host_fetch(const int p, const Coord& cox);

/**
 * @internal
 *
 * Set node values as 128-bit SSE value in main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @param p First trajectory id.
 * @param cox Base coordinates.
 *
 * @param val Value to set for the given node.
 */
template<class B, class X, int Xo, int Yo, int Zo>
void sse_host_put(const int p, const Coord& cox,
    const sse_real& val);

/**
 * @internal
 *
 * Facade for state as 128-bit SSE values in main memory.
 *
 * @ingroup state_host
 */
struct sse_host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  BI_FORCE_INLINE static sse_real fetch(const int p, const Coord& cox) {
    return sse_host_fetch<B,X,Xo,Yo,Zo>(p, cox);
  }

  /**
   * Put value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  BI_FORCE_INLINE static void put(const int p, const Coord& cox,
      const sse_real& val) {
    sse_host_put<B,X,Xo,Yo,Zo>(p, cox, val);
  }
};

}

#include "../math/sse_state.hpp"

template<class B, class X, int Xo, int Yo, int Zo>
BI_FORCE_INLINE inline bi::sse_real bi::sse_host_fetch(const int p,
    const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();
  sse_real result;

  if (is_s_node<X>::value) {
    result = sse_state_get(hostSState, 0, i);
  } else if (is_d_node<X>::value) {
    result = sse_state_get(hostDState, p, i);
  } else if (is_c_node<X>::value) {
    result = sse_state_get(hostCState, p, i);
  } else if (is_r_node<X>::value) {
    result = sse_state_get(hostRState, p, i);
  } else if (is_f_node<X>::value) {
    result = sse_state_get(hostFState, 0, i);
  } else if (is_o_node<X>::value) {
    result = sse_state_get(hostOYState, 0, i);
  } else if (is_p_node<X>::value) {
    result = sse_state_get(hostPState, 0, i);
  }
  return result;
}

template<class B, class X, int Xo, int Yo, int Zo>
BI_FORCE_INLINE inline void bi::sse_host_put(const int p,
    const Coord& cox, const sse_real& val) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();

  if (is_s_node<X>::value) {
    sse_state_set(hostSState, 0, i, val);
  } else if (is_d_node<X>::value) {
    sse_state_set(hostDState, p, i, val);
  } else if (is_c_node<X>::value) {
    sse_state_set(hostCState, p, i, val);
  } else if (is_r_node<X>::value) {
    sse_state_set(hostRState, p, i, val);
  } else if (is_f_node<X>::value) {
    sse_state_set(hostFState, 0, i, val);
  } else if (is_o_node<X>::value) {
    sse_state_set(hostOYState, 0, i, val);
  } else if (is_p_node<X>::value) {
    sse_state_set(hostPState, 0, i, val);
  }
}

#endif

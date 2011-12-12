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
#ifndef BI_HOST_SSE_CONST_HOST_HPP
#define BI_HOST_SSE_CONST_HOST_HPP

#include "../math/sse.hpp"
#include "../misc/macro.hpp"
#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Fetch node values from constant main memory as 128-bit SSE value.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @param cox Base coordinates.
 *
 * @return 2- (double precision) or 4- (single precision) component packed
 * vector of node values beginning at trajectory @p p.
 */
template<class B, class X, int Xo, int Yo, int Zo>
sse_real sse_const_host_fetch(const Coord& cox);

/**
 * @internal
 *
 * Facade for state as 128-bit SSE values in constant main memory.
 *
 * @ingroup state_host
 */
struct sse_const_host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  BI_FORCE_INLINE static sse_real fetch(const int p, const Coord& cox) {
    return sse_const_host_fetch<B,X,Xo,Yo,Zo>(cox);
  }
};

}

#include "../math/sse_state.hpp"

template<class B, class X, int Xo, int Yo, int Zo>
BI_FORCE_INLINE inline bi::sse_real bi::sse_const_host_fetch(
    const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();
  sse_real result;

  if (is_s_node<X>::value) {
    result = sse_state_get(constHostSState, 0, i);
  } else if (is_p_node<X>::value) {
    result = sse_state_get(constHostPState, 0, i);
  } else {
    BI_ASSERT(false, "Unsupported node type");
  }
  return result;
}

#endif

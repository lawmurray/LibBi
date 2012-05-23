/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_BIND_HPP
#define BI_HOST_BIND_HPP

#include "host.hpp"
#include "const_host.hpp"

namespace bi {
/**
 * @internal
 *
 * Bind state to global variables.
 *
 * @tparam B Model type.
 *
 * @param s State.
 *
 * The use of global variables for holding the state is motivated by the
 * following:
 *
 * @li @c __constant__ and @c __texture__ references in CUDA must be global
 * variables, and we wish to position certain elements of the state in these
 * memory types,
 * @li doing so saves repeated transfer of common kernel arguments to the
 * device and has been observed to reduce register usage in some
 * circumstances,
 * @li we have chosen to likewise create global variables for ordinary host
 * and device memory, to have a common interface for retrieving required
 * data regardless of memory type.
 */
template<class B>
void bind(State<B,ON_HOST>& s);

/**
 * @internal
 *
 * @copydoc bind(State<B,ON_HOST>&)
 */
template<class B>
void bind(State<B,ON_DEVICE>& s);

/**
 * @internal
 *
 * Unbind state from global variables.
 */
template<class B>
void unbind(const State<B,ON_HOST>& s);

/**
 * @internal
 *
 * Unbind state from global variables.
 */
template<class B>
void unbind(const State<B,ON_DEVICE>& s);

}

template<class B>
inline void bi::bind(State<B,ON_HOST>& s) {
  host_bind_r(s.get(R_VAR));
  host_bind_d(s.get(D_VAR));
  host_bind_p(s.get(P_VAR));
  host_bind_f(s.get(F_VAR));
  host_bind_o(s.get(O_VAR));
  host_bind_px(s.get(PX_VAR));
  host_bind_dx(s.get(DX_VAR));
  host_bind_ry(s.getAlt(R_VAR));
  host_bind_dy(s.getAlt(D_VAR));
  host_bind_py(s.getAlt(P_VAR));
  host_bind_oy(s.getAlt(O_VAR));

  const_host_bind_p(s.get(P_VAR));
  const_host_bind_px(s.get(PX_VAR));
}

template<class B>
inline void bi::unbind(const State<B,ON_HOST>& s) {
  //
}

#ifdef __CUDACC__
#include "../cuda/bind.cuh"
#endif

#endif

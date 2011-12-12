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
void bind(State<ON_HOST>& s);

/**
 * @internal
 *
 * @copydoc bind(State<ON_HOST>&)
 */
void bind(State<ON_DEVICE>& s);

/**
 * @internal
 *
 * Bind static state to global variables.
 */
void bind(Static<ON_HOST>& theta);

/**
 * @internal
 *
 * @copydoc bind(Static<ON_HOST>&)
 */
void bind(Static<ON_DEVICE>& theta);

/**
 * @internal
 *
 * Unbind state from global variables.
 */
void unbind(const State<ON_HOST>& s);

/**
 * @internal
 *
 * Unbind state from global variables.
 */
void unbind(const State<ON_DEVICE>& s);

/**
 * @internal
 *
 * Unbind static state from global variables.
 */
void unbind(const Static<ON_HOST>& theta);

/**
 * @internal
 *
 * @copydoc unbind(Static<ON_HOST>&)
 */
void unbind(const Static<ON_DEVICE>& theta);

}

inline void bi::bind(State<ON_HOST>& s) {
  host_bind_d(s.get(D_NODE));
  host_bind_c(s.get(C_NODE));
  host_bind_r(s.get(R_NODE));
  host_bind_f(s.get(F_NODE));
  host_bind_o(s.get(O_NODE));
  host_bind_oy(s.get(OY_NODE));
  host_bind_or(s.get(OR_NODE));
}

inline void bi::bind(Static<ON_HOST>& theta) {
  host_bind_s(theta.get(S_NODE));
  host_bind_p(theta.get(P_NODE));

  if (theta.size() == 1) {
    const_host_bind_s(theta.get(S_NODE));
    const_host_bind_p(theta.get(P_NODE));
  }
}

inline void bi::unbind(const State<ON_HOST>& s) {
  //
}

inline void bi::unbind(const Static<ON_HOST>& theta) {
  //
}

#ifdef __CUDACC__
#include "../cuda/bind.cuh"
#endif

#endif

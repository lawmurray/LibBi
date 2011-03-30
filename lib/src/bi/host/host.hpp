/**
 * @file
 *
 * Functions for reading of state objects through main memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_HOST_HPP
#define BI_HOST_HOST_HPP

#include "../misc/macro.hpp"
#include "../state/Coord.hpp"

/**
 * @internal
 *
 * @def HOST_STATE_DEC
 *
 * Macro for global state variable declarations in main memory.
 */
#define HOST_STATE_DEC(name, Name) \
  /**
    Main memory name##-state.
   */ \
  static bi::host_matrix_handle<real> host##Name##State;

HOST_STATE_DEC(s, S)
HOST_STATE_DEC(d, D)
HOST_STATE_DEC(c, C)
HOST_STATE_DEC(r, R)
HOST_STATE_DEC(f, F)
HOST_STATE_DEC(o, O)
HOST_STATE_DEC(p, P)
HOST_STATE_DEC(oy, OY)
HOST_STATE_DEC(or, OR)

namespace bi {
/**
 * @internal
 *
 * @def HOST_BIND_DEC
 *
 * Macro for bind function declarations.
 */
#define HOST_BIND_DEC(name) \
  /**
    @internal

    Bind name##-net state to main memory.

    @ingroup state_host

    @param s State.
   */ \
   CUDA_FUNC_HOST void host_bind_##name(host_matrix_reference<real>& s);

HOST_BIND_DEC(s)
HOST_BIND_DEC(d)
HOST_BIND_DEC(c)
HOST_BIND_DEC(r)
HOST_BIND_DEC(f)
HOST_BIND_DEC(o)
HOST_BIND_DEC(p)
HOST_BIND_DEC(oy)
HOST_BIND_DEC(or)

/**
 * @internal
 *
 * Fetch node value from main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @param p Trajectory id. Ignored for f- and oy-node requests, as only one
 * trajectory is ever stored.
 * @param cox Base coordinates.
 *
 * @return Value of the given node.
 */
template<class B, class X, int Xo, int Yo, int Zo>
real host_fetch(const int p, const Coord& cox);

/**
 * @internal
 *
 * Set node value in main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @param p Trajectory id. Ignored for f- and oy-node requests, as only one
 * trajectory is ever stored.
 * @param cox Base coordinates.
 * @param val Value to set for the given node.
 */
template<class B, class X, int Xo, int Yo, int Zo>
void host_put(const int p, const Coord& cox, const real& val);

/**
 * @internal
 *
 * Facade for state in main memory.
 *
 * @ingroup state_host
 */
struct host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  static real fetch(const int p, const Coord& cox) {
    return host_fetch<B,X,Xo,Yo,Zo>(p, cox);
  }

  /**
   * Put value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  static void put(const int p, const Coord& cox, const real& val) {
    host_put<B,X,Xo,Yo,Zo>(p, cox, val);
  }
};

}

/**
 * @internal
 *
 * @def HOST_BIND_DEF
 *
 * Macro for bind function definitions.
 */
#define HOST_BIND_DEF(name, Name) \
  inline void bi::host_bind_##name(host_matrix_reference<real>& s) { \
    host##Name##State.copy(s); \
  }

HOST_BIND_DEF(s, S)
HOST_BIND_DEF(d, D)
HOST_BIND_DEF(c, C)
HOST_BIND_DEF(r, R)
HOST_BIND_DEF(f, F)
HOST_BIND_DEF(o, O)
HOST_BIND_DEF(p, P)
HOST_BIND_DEF(oy, OY)
HOST_BIND_DEF(or, OR)

template<class B, class X, int Xo, int Yo, int Zo>
inline real bi::host_fetch(const int p, const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();
  real result;

  if (is_s_node<X>::value) {
    result = hostSState(p, i);
  } else if (is_d_node<X>::value) {
    result = hostDState(p, i);
  } else if (is_c_node<X>::value) {
    result = hostCState(p, i);
  } else if (is_r_node<X>::value) {
    result = hostRState(p, i);
  } else if (is_f_node<X>::value) {
    result = hostFState(0, i);
  } else if (is_o_node<X>::value) {
    result = hostOYState(0, i);
  } else if (is_p_node<X>::value) {
    result = hostPState(p, i);
  }
  return result;
}

template<class B, class X, int Xo, int Yo, int Zo>
inline void bi::host_put(const int p, const Coord& cox,
    const real& val) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();

  if (is_s_node<X>::value) {
    hostSState(p, i) = val;
  } else if (is_d_node<X>::value) {
    hostDState(p, i) = val;
  } else if (is_c_node<X>::value) {
    hostCState(p, i) = val;
  } else if (is_r_node<X>::value) {
    hostRState(p, i) = val;
  } else if (is_f_node<X>::value) {
    hostFState(0, i) = val;
  } else if (is_o_node<X>::value) {
    hostOYState(0, i) = val;
  } else if (is_p_node<X>::value) {
    hostPState(p, i) = val;
  }
}

#endif

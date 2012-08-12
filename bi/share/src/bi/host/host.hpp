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

#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../misc/macro.hpp"
#include "../traits/var_traits.hpp"

/**
 * @internal
 *
 * @def HOST_STATE_DEC
 *
 * Macro for declaring host global variable for input buffer.
 */
#define HOST_STATE_DEC(name, Name) \
  /**
    Main memory name##-state input.
   */ \
  static bi::host_matrix_handle<real> host##Name##State;

HOST_STATE_DEC(r, R)
HOST_STATE_DEC(d, D)
HOST_STATE_DEC(p, P)
HOST_STATE_DEC(f, F)
HOST_STATE_DEC(o, O)
HOST_STATE_DEC(dx, DX)
HOST_STATE_DEC(px, PX)
HOST_STATE_DEC(ry, RY)
HOST_STATE_DEC(dy, DY)
HOST_STATE_DEC(py, PY)
HOST_STATE_DEC(oy, OY)

namespace bi {
/**
 * @internal
 *
 * @def HOST_BIND_DEC
 *
 * Macro for declaring bind function for node type.
 */
#define HOST_BIND_DEC(name) \
  /**
    Bind name##-net input buffer to main memory.

    @ingroup state_host

    @param s State.
   */ \
   CUDA_FUNC_HOST void host_bind_##name(host_matrix_reference<real> s);

HOST_BIND_DEC(r)
HOST_BIND_DEC(d)
HOST_BIND_DEC(p)
HOST_BIND_DEC(f)
HOST_BIND_DEC(o)
HOST_BIND_DEC(dx)
HOST_BIND_DEC(px)
HOST_BIND_DEC(ry)
HOST_BIND_DEC(dy)
HOST_BIND_DEC(py)
HOST_BIND_DEC(oy)

/**
 * Fetch node value from main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p Trajectory id.
 * @param ix Serial coordinate.
 *
 * @return Value of the given node.
 */
template<class B, class X>
real host_fetch(const int p, const int ix);

/**
 * Fetch node value from alternative buffer in main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p Trajectory id.
 * @param ix Serial coordinate.
 *
 * @return Value of the given node.
 */
template<class B, class X>
real host_fetch_alt(const int p, const int ix);

/**
 * Set node value in main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p Trajectory id.
 * @param ix Serial coordinate.
 * @param val Value to set for the given node.
 */
template<class B, class X>
void host_put(const int p, const int ix, const real& val);

/**
 * Facade for state in main memory.
 *
 * @ingroup state_host
 */
struct host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X>
  static real fetch(const int p, const int ix) {
    return host_fetch<B,X>(p, ix);
  }

  /**
   * Fetch alternative value.
   */
  template<class B, class X>
  static real fetch_alt(const int p, const int ix) {
    return host_fetch_alt<B,X>(p, ix);
  }

  /**
   * Put value.
   */
  template<class B, class X>
  static void put(const int p, const int ix, const real& val) {
    host_put<B,X>(p, ix, val);
  }
};

}

/**
 * @internal
 *
 * @def HOST_BIND_DEF
 *
 * Macro for defining bind function for node type.
 */
#define HOST_BIND_DEF(name, Name) \
  inline void bi::host_bind_##name(host_matrix_reference<real> s) { \
    host##Name##State.copy(s); \
  }

HOST_BIND_DEF(r, R)
HOST_BIND_DEF(d, D)
HOST_BIND_DEF(p, P)
HOST_BIND_DEF(f, F)
HOST_BIND_DEF(o, O)
HOST_BIND_DEF(dx, DX)
HOST_BIND_DEF(px, PX)
HOST_BIND_DEF(ry, RY)
HOST_BIND_DEF(dy, DY)
HOST_BIND_DEF(py, PY)
HOST_BIND_DEF(oy, OY)

template<class B, class X>
inline real bi::host_fetch(const int p, const int ix) {
  const int i = var_net_start<B,X>::value + ix;
  real result;

  if (is_d_var<X>::value) {
    result = hostDState(p, i);
  } else if (is_dx_var<X>::value) {
    result = hostDXState(p, i);
  } else if (is_r_var<X>::value) {
    result = hostRState(p, i);
  } else if (is_f_var<X>::value) {
    result = hostFState(0, i);
  } else if (is_o_var<X>::value) {
    result = hostOState(p, i);
  } else if (is_p_var<X>::value) {
    result = hostPState(0, i);
  } else if (is_px_var<X>::value) {
    result = hostPXState(0, i);
  }
  return result;
}

template<class B, class X>
inline real bi::host_fetch_alt(const int p, const int ix) {
  const int i = var_net_start<B,X>::value + ix;
  real result;

  if (is_d_var<X>::value) {
    result = hostDYState(p, i);
  } else if (is_r_var<X>::value) {
    result = hostRYState(p, i);
  } else if (is_o_var<X>::value) {
    result = hostOYState(0, i);
  } else if (is_p_var<X>::value) {
    result = hostPYState(0, i);
  }
  return result;
}

template<class B, class X>
inline void bi::host_put(const int p, const int ix, const real& val) {
  const int i = var_net_start<B,X>::value + ix;

  if (is_d_var<X>::value) {
    hostDState(p, i) = val;
  } else if (is_dx_var<X>::value) {
    hostDXState(p, i) = val;
  } else if (is_r_var<X>::value) {
    hostRState(p, i) = val;
  } else if (is_f_var<X>::value) {
    hostFState(0, i) = val;
  } else if (is_o_var<X>::value) {
    hostOState(p, i) = val;
  } else if (is_p_var<X>::value) {
    hostPState(0, i) = val;
  } else if (is_px_var<X>::value) {
    hostPXState(0, i) = val;
  }
}

#endif

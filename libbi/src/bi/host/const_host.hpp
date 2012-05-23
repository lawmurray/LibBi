/**
 * @file
 *
 * Functions for reading of single-trajectory state objects through main
 * memory. Analogue of constant.cuh for device memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * @see constant.cuh
 */
#ifndef BI_HOST_CONSTHOST_HPP
#define BI_HOST_CONSTHOST_HPP

#include "../misc/macro.hpp"

/**
 * @internal
 *
 * @def CONST_HOST_STATE_DEC
 *
 * Macro for constant state variable declarations in main memory.
 */
#define CONST_HOST_STATE_DEC(name, Name) \
  /**
    Main memory name##-state.
   */ \
  static bi::host_matrix_handle<real> constHost##Name##State;

CONST_HOST_STATE_DEC(p, P)
CONST_HOST_STATE_DEC(px, PX)

namespace bi {
/**
 * @internal
 *
 * @def CONST_HOST_BIND_DEC
 *
 * Macro for bind function declarations.
 */
#define CONST_HOST_BIND_DEC(name) \
  /**
    Bind name##-net state to main memory.

    @ingroup state_host

    @param s State.
   */ \
   CUDA_FUNC_HOST void const_host_bind_##name(host_matrix_reference<real> s);

CONST_HOST_BIND_DEC(p)
CONST_HOST_BIND_DEC(px)

/**
 * Fetch node value from constant main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param ix Serial coordinate.
 *
 * @return Value of the given node.
 */
template<class B, class X>
real const_host_fetch(const int ix);

/**
 * Fetch alternative node value from constant main memory.
 *
 * @ingroup state_host
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param ix Serial coordinate.
 *
 * @return Value of the given node.
 */
template<class B, class X>
real const_host_fetch_alt(const int ix);

/**
 * Facade for state in constant main memory.
 *
 * @ingroup state_host
 */
struct const_host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X>
  static real fetch(const int p, const int ix) {
    return const_host_fetch<B,X>(ix);
  }

  /**
   * Fetch alternative value.
   */
  template<class B, class X>
  static real fetch_alt(const int p, const int ix) {
    return host_fetch_alt<B,X>(p, ix);
  }

};

}

/**
 * @internal
 *
 * @def CONST_HOST_BIND_DEF
 *
 * Macro for bind function definitions.
 */
#define CONST_HOST_BIND_DEF(name, Name) \
  inline void bi::const_host_bind_##name(host_matrix_reference<real> s) { \
    constHost##Name##State.copy(s); \
  }

CONST_HOST_BIND_DEF(p, P)
CONST_HOST_BIND_DEF(px, PX)

template<class B, class X>
inline real bi::const_host_fetch(const int ix) {
  const int i = var_net_start<B,X>::value + ix;
  real result;

  if (is_p_var<X>::value) {
    result = constHostPState(0, i);
  } else if (is_px_var<X>::value) {
    result = constHostPXState(0, i);
  } else {
    BI_ASSERT(false, "Unsupported node type");
  }
  return result;
}

template<class B, class X>
inline real bi::const_host_fetch_alt(const int ix) {
  const int i = var_net_start<B,X>::value + ix;
  real result;

  return result;
}

#endif

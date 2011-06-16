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
#ifndef BI_HOST_CONST_HOST_HPP
#define BI_HOST_CONST_HOST_HPP

#include "../misc/macro.hpp"
#include "../state/Coord.hpp"

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

CONST_HOST_STATE_DEC(s, S)
CONST_HOST_STATE_DEC(p, P)

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
    @internal

    Bind name##-net state to main memory.

    @ingroup state_host

    @param s State.
   */ \
   CUDA_FUNC_HOST void const_host_bind_##name(host_matrix_reference<real>& s);

CONST_HOST_BIND_DEC(s)
CONST_HOST_BIND_DEC(p)

/**
 * @internal
 *
 * Fetch node value from constant main memory.
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
 * @return Value of the given node.
 */
template<class B, class X, int Xo, int Yo, int Zo>
real const_host_fetch(const Coord& cox);

/**
 * @internal
 *
 * Facade for state in constant main memory.
 *
 * @ingroup state_host
 */
struct const_host {
  static const bool on_device = false;

  /**
   * Fetch value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  static real fetch(const int p, const Coord& cox) {
    return const_host_fetch<B,X,Xo,Yo,Zo>(cox);
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
  inline void bi::const_host_bind_##name(host_matrix_reference<real>& s) { \
    constHost##Name##State.copy(s); \
  }

CONST_HOST_BIND_DEF(s, S)
CONST_HOST_BIND_DEF(p, P)

template<class B, class X, int Xo, int Yo, int Zo>
inline real bi::const_host_fetch(const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();
  real result;

  if (is_s_node<X>::value) {
    result = constHostSState(0, i);
  } else if (is_p_node<X>::value) {
    result = constHostPState(0, i);
  } else {
    BI_ASSERT(false, "Unsupported node type");
  }
  return result;
}

#endif

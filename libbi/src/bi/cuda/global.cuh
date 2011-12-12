/**
 * @file
 *
 * Functions for reading of state objects through global memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_GLOBAL_CUH
#define BI_CUDA_GLOBAL_CUH

#include "constant.cuh"
#include "../misc/macro.hpp"
#include "../state/Coord.hpp"

/**
 * @internal
 *
 * @def GLOBAL_STATE_DEC
 *
 * Macro for global state variable declarations.
 */
#define GLOBAL_STATE_DEC(name, Name) \
  /**
    @internal

    Global memory name##-state.
   */ \
  static CUDA_VAR_CONSTANT bi::gpu_matrix_handle<real> global##Name##State; \
  \
  /**
    @internal

    Host-side guard for writing to #global##Name##State, saves host-to-device
    copy when unchanged.
   */ \
  static bi::gpu_matrix_handle<real> guard##Name##State;

GLOBAL_STATE_DEC(s, S)
GLOBAL_STATE_DEC(d, D)
GLOBAL_STATE_DEC(c, C)
GLOBAL_STATE_DEC(r, R)
GLOBAL_STATE_DEC(f, F)
GLOBAL_STATE_DEC(o, O)
GLOBAL_STATE_DEC(p, P)
GLOBAL_STATE_DEC(oy, OY)
GLOBAL_STATE_DEC(or, OR)

namespace bi {
/**
 * @internal
 *
 * @def GLOBAL_BIND_DEC
 *
 * Macro for bind function declarations.
 */
#define GLOBAL_BIND_DEC(name) \
  /**
    @internal

    Bind name##-net state to global memory.

    @ingroup state_gpu

    @param s State.
   */ \
   template<class M1> \
   CUDA_FUNC_HOST void global_bind_##name(M1& s);

GLOBAL_BIND_DEC(s)
GLOBAL_BIND_DEC(d)
GLOBAL_BIND_DEC(c)
GLOBAL_BIND_DEC(r)
GLOBAL_BIND_DEC(f)
GLOBAL_BIND_DEC(o)
GLOBAL_BIND_DEC(p)
GLOBAL_BIND_DEC(oy)
GLOBAL_BIND_DEC(or)

/**
 * @internal
 *
 * Fetch node value from global memory.
 *
 * @ingroup state_gpu
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
CUDA_FUNC_DEVICE real global_fetch(const int p, const Coord& cox);

/**
 * @internal
 *
 * Set node value in global memory.
 *
 * @ingroup state_gpu
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
 * @param val Value to set for the given node.
 */
template<class B, class X, int Xo, int Yo, int Zo>
CUDA_FUNC_DEVICE void global_put(const int p, const Coord& cox,
    const real& val);

/**
 * @internal
 *
 * Facade for state in global memory.
 *
 * @ingroup state_gpu
 */
struct global {
  static const bool on_device = true;

  /**
   * Fetch value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  static CUDA_FUNC_DEVICE real fetch(const int p, const Coord& cox) {
    return global_fetch<B,X,Xo,Yo,Zo>(p, cox);
  }

  /**
   * Put value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  static CUDA_FUNC_DEVICE void put(const int p, const Coord& cox,
      const real& val) {
    global_put<B,X,Xo,Yo,Zo>(p, cox, val);
  }
};

}

/**
 * @internal
 *
 * @def GLOBAL_BIND_DEF
 *
 * Macro for bind function definitions.
 *
 * @note bind methods are inlined due to implied static storage of
 * __constant__ vars.
 */
#define GLOBAL_BIND_DEF(name, Name) \
  template<class M1> \
  inline void bi::global_bind_##name(M1& s) { \
    if (s.size1()*s.size2() > 0 && !guard##Name##State.same(s)) { \
      CUDA_SET_CONSTANT(gpu_matrix_handle<real>, \
          MACRO_QUOTE(global##Name##State), s); \
      guard##Name##State.copy(s); \
    } \
  }

GLOBAL_BIND_DEF(s, S)
GLOBAL_BIND_DEF(d, D)
GLOBAL_BIND_DEF(c, C)
GLOBAL_BIND_DEF(r, R)
GLOBAL_BIND_DEF(f, F)
GLOBAL_BIND_DEF(o, O)
GLOBAL_BIND_DEF(p, P)
GLOBAL_BIND_DEF(oy, OY)
GLOBAL_BIND_DEF(or, OR)

template<class B, class X, int Xo, int Yo, int Zo>
inline real bi::global_fetch(const int p, const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();
  real result;

  if (is_s_node<X>::value) {
    result = globalSState(p, i);
  } else if (is_d_node<X>::value) {
    result = globalDState(p, i);
  } else if (is_c_node<X>::value) {
    result = globalCState(p, i);
  } else if (is_r_node<X>::value) {
    result = globalRState(p, i);
  } else if (is_f_node<X>::value) {
    result = globalFState(0, i);
  } else if (is_o_node<X>::value) {
    result = globalOYState(0, i);
  } else if (is_p_node<X>::value) {
    result = globalPState(p, i);
  } else {
    result = REAL(1.0 / 0.0);
  }
  return result;
}

template<class B, class X, int Xo, int Yo, int Zo>
inline void bi::global_put(const int p, const Coord& cox,
    const real& val) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();

  if (is_s_node<X>::value) {
    globalSState(p, i) = val;
  } else if (is_d_node<X>::value) {
    globalDState(p, i) = val;
  } else if (is_c_node<X>::value) {
    globalCState(p, i) = val;
  } else if (is_r_node<X>::value) {
    globalRState(p, i) = val;
  } else if (is_f_node<X>::value) {
    globalFState(0, i) = val;
  } else if (is_o_node<X>::value) {
    globalOYState(0, i) = val;
  } else if (is_p_node<X>::value) {
    globalPState(p, i) = val;
  }
}

#endif

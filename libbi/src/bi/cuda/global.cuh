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

/**
 * @internal
 *
 * @def GLOBAL_STATE_DEC
 *
 * Macro for declaring device global variable for input buffer.
 */
#define GLOBAL_STATE_DEC(name, Name) \
  /**
    Global memory name##-state input.
   */ \
  static CUDA_VAR_CONSTANT bi::gpu_matrix_handle<real> global##Name##State; \
  \
  /**
    Host-side guard for writing to #global##Name##State, saves
    host-to-device copy when unchanged.
   */ \
  static bi::gpu_matrix_handle<real> guard##Name##State;

GLOBAL_STATE_DEC(d, D)
GLOBAL_STATE_DEC(dx, DX)
GLOBAL_STATE_DEC(r, R)
GLOBAL_STATE_DEC(f, F)
GLOBAL_STATE_DEC(o, O)
GLOBAL_STATE_DEC(p, P)
GLOBAL_STATE_DEC(px, PX)

namespace bi {
/**
 * @internal
 *
 * @def GLOBAL_BIND_DEC
 *
 * Macro for declaring bind function for node type.
 */
#define GLOBAL_BIND_DEC(name) \
  /**
    @internal

    Bind name##-net input buffer to global memory.

    @ingroup state_gpu

    @param s State.
   */ \
   template<class M1> \
   CUDA_FUNC_HOST void global_bind_##name(M1 s);

GLOBAL_BIND_DEC(d)
GLOBAL_BIND_DEC(dx)
GLOBAL_BIND_DEC(r)
GLOBAL_BIND_DEC(f)
GLOBAL_BIND_DEC(o)
GLOBAL_BIND_DEC(p)
GLOBAL_BIND_DEC(px)

/**
 * Fetch node value from global memory.
 *
 * @ingroup state_gpu
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p Trajectory id.
 * @param ix Serial coordinate.
 *
 * @return Value.
 */
template<class B, class X>
CUDA_FUNC_DEVICE real global_fetch(const int p, const int ix);

/**
 * Set node value in global memory.
 *
 * @ingroup state_gpu
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p Trajectory id.
 * @param ix Serial coordinate.
 * @param val Value to set.
 */
template<class B, class X>
CUDA_FUNC_DEVICE void global_put(const int p, const int ix, const real& val);

/**
 * Facade for state in global memory.
 *
 * @ingroup state_gpu
 */
struct global {
  static const bool on_device = true;

  /**
   * Fetch value.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE real fetch(const int p, const int ix) {
    return global_fetch<B,X>(p, ix);
  }

  /**
   * Put value.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE void put(const int p, const int ix,
      const real& val) {
    global_put<B,X>(p, ix, val);
  }
};

}

/**
 * @internal
 *
 * @def GLOBAL_BIND_DEF
 *
 * Macro for defining bind function for node type.
 *
 * @note These bind methods must be inlined due to implied static storage of
 * __constant__ vars.
 */
#define GLOBAL_BIND_DEF(name, Name) \
  template<class M1> \
  inline void bi::global_bind_##name(M1 s) { \
    if (s.size1()*s.size2() > 0 && !guard##Name##State.same(s)) { \
      CUDA_SET_CONSTANT(gpu_matrix_handle<real>, \
          MACRO_QUOTE(global##Name##State), s); \
      guard##Name##State.copy(s); \
    } \
  }

GLOBAL_BIND_DEF(d, D)
GLOBAL_BIND_DEF(dx, DX)
GLOBAL_BIND_DEF(r, R)
GLOBAL_BIND_DEF(f, F)
GLOBAL_BIND_DEF(o, O)
GLOBAL_BIND_DEF(p, P)
GLOBAL_BIND_DEF(px, PX)

template<class B, class X>
inline real bi::global_fetch(const int p, const int ix) {
  const int i = var_net_start<B,X>::value + ix;
  real result;

  if (is_d_var<X>::value) {
    result = globalDState(p, i);
  } else if (is_dx_var<X>::value) {
    result = globalDXState(p, i);
  } else if (is_r_var<X>::value) {
    result = globalRState(p, i);
  } else if (is_f_var<X>::value) {
    result = globalFState(0, i);
  } else if (is_o_var<X>::value) {
    result = globalOState(p, i);
  } else if (is_p_var<X>::value) {
    result = globalPState(0, i);
  } else if (is_px_var<X>::value) {
    result = globalPXState(0, i);
  } else {
    result = BI_REAL(1.0/0.0);
  }
  return result;
}

template<class B, class X>
inline void bi::global_put(const int p, const int ix, const real& val) {
  const int i = var_net_start<B,X>::value + ix;

  if (is_d_var<X>::value) {
    globalDState(p, i) = val;
  } else if (is_dx_var<X>::value) {
    globalDXState(p, i) = val;
  } else if (is_r_var<X>::value) {
    globalRState(p, i) = val;
  } else if (is_f_var<X>::value) {
    globalFState(0, i) = val;
  } else if (is_o_var<X>::value) {
    globalOState(p, i) = val;
  } else if (is_p_var<X>::value) {
    globalPState(0, i) = val;
  } else if (is_px_var<X>::value) {
    globalPXState(0, i) = val;
  }
}

#endif

/**
 * @file
 *
 * Functions for efficient reading of model and state objects through
 * constant memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Constant memory may be used for storing model parent structures, and for
 * storing states which are constant for all trajectories -- f-, o- and
 * p-nodes states.
 */
#ifndef BI_CUDA_CONSTANT_CUH
#define BI_CUDA_CONSTANT_CUH

#include "cuda.hpp"
#include "../model/model.hpp"
#include "../misc/macro.hpp"
#include "../state/Coord.hpp"

/**
 * @def CONST_STATE_SIZE
 *
 * Number of reals allocated to store states in %constant memory
 */
#define CONST_STATE_SIZE 128

/*
 * Constant memory buffers.
 */
static CUDA_VAR_CONSTANT real const_mem_p[CONST_STATE_SIZE];
static CUDA_VAR_CONSTANT real const_mem_s[CONST_STATE_SIZE];

namespace bi {
/**
 * @internal
 *
 * @def CONSTANT_BIND_DEC
 *
 * Macro for functions binding state to constant memory. State on device
 * is copied to constant memory on device.
 */
#define CONSTANT_BIND_DEC(name) \
  /**
    @internal

    Bind state to %constant memory.

    @ingroup state_gpu
   */ \
  template<class M1> \
  CUDA_FUNC_HOST void const_bind_##name(const M1& s);

CONSTANT_BIND_DEC(s)
CONSTANT_BIND_DEC(p)
/**
 * @internal
 *
 * Fetch node value from %constant memory.
 *
 * @ingroup state_gpu
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
 *
 * Note that only one trajectory is ever stored in constant memory, so no
 * @c p argument is given, unlike global_fetch().
 */
template<class B, class X, int Xo, int Yo, int Zo>
CUDA_FUNC_DEVICE real const_fetch(const Coord& cox);

/**
 * @internal
 *
 * Facade for node state in %constant memory.
 *
 * @ingroup state_gpu
 */
struct constant {
  static const bool on_device = true;

  /**
   * Fetch value.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  static CUDA_FUNC_DEVICE real fetch(const int p, const Coord& cox) {
    return const_fetch<B,X,Xo,Yo,Zo>(cox);
  }
};

}

/**
 * @internal
 *
 * @def CONSTANT_BIND_DEF
 *
 * Macro for %constant memory %bind function definitions.
 */
#define CONSTANT_BIND_DEF(name, Name) \
  template<class M1> \
  inline void bi::const_bind_##name(const M1& s) { \
    /* pre-condition */ \
    assert (M1::on_device); \
    assert (s.size1() == 1 && s.lead() == 1); \
    \
    CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync( \
        MACRO_QUOTE(const_mem_##name), s.buf(), s.size2()*sizeof(real), \
        0, cudaMemcpyDeviceToDevice, 0)); \
  }

CONSTANT_BIND_DEF(s, S)
CONSTANT_BIND_DEF(p, P)

template<class B, class X, int Xo, int Yo, int Zo>
inline real bi::const_fetch(const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();

  if (is_p_node<X>::value) {
    return const_mem_p[i];
  } else if (is_s_node<X>::value) {
    return const_mem_s[i];
  } else {
    return REAL(1.0 / 0.0);
  }
}

/**
 * @internal
 *
 * Number of trajectories, in %constant memory.
 */
static CUDA_VAR_CONSTANT int constP;

/**
 * @internal
 *
 * Host-side guard for writing to #constP, saves host-to-device copy when
 * unchanged.
 */
static int guardP = -1;

namespace bi {
/**
 * @internal
 *
 * Bind number of trajectories to %constant memory.
 *
 * @ingroup state_gpu
 *
 * @param P Number of trajectories.
 */
CUDA_FUNC_HOST void const_bind(const int P);

}

inline void bi::const_bind(const int P) {
  if (P != guardP) {
    CUDA_SET_CONSTANT(int, MACRO_QUOTE(constP), P);
    guardP = P;
  }
}

#endif

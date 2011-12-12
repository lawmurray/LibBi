/**
 * @file
 *
 * Functions for manipulating state objects in shared memory.

 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Each thread executed on the GPU is associated with one node (across y) in
 * one trajectory (across x). These are implicitly calculated from the
 * execution configuration and thread ids.
 *
 * Shared memory is currently supported only for c-nodes and d-nodes, and not
 * simultaneously.
 *
 * Some of these functions rely on constant memory also and should be used in
 * conjunction with those in constant.cuh. First bind a model object to the
 * global constant memory variables using constant_bind() on the host, then
 * initialise shared memory using shared_init() on the device. Any of
 * the shared memory fetch and put functions may then be used on the device
 * to read and write data to shared rather than global memory. At completion,
 * call shared_commit() on the device to commit the changes to shared memory
 * back to global memory, and then constant_unbind() on the host.
 */
#ifndef BI_CUDA_SHARED_CUH
#define BI_CUDA_SHARED_CUH

#include "cuda.hpp"
#include "../state/Coord.hpp"

/**
 * @internal
 *
 * Shared memory.
 */
extern CUDA_VAR_SHARED real shared_mem[];

namespace bi {
/**
 * @internal
 *
 * @def SHARED_INIT_DEC
 *
 * Macro for shared memory init function declarations.
 */
#define SHARED_INIT_DEC(name) \
  /**
    @internal

    Copy state into shared memory.
   */ \
  CUDA_FUNC_DEVICE void shared_init_##name();

SHARED_INIT_DEC(c)
SHARED_INIT_DEC(d)

/**
 * @internal
 *
 * @def SHARED_COMMIT_DEC
 *
 * Macro for shared memory commit function declarations.
 */
#define SHARED_COMMIT_DEC(name) \
  /**
    @internal

    Commit changes in shared memory back to state.
   */ \
  CUDA_FUNC_DEVICE void shared_commit_##name();

SHARED_COMMIT_DEC(c)
SHARED_COMMIT_DEC(d)

/**
 * @internal
 *
 * Fetch node value from shared memory.
 *
 * @ingroup state_gpu
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @return Value of the node pertaining to the current thread, in the
 * trajectory pertaining to the current thread.
 */
template<class B, class X, int Xo, int Yo, int Zo>
CUDA_FUNC_DEVICE real shared_fetch(const Coord& cox);

/**
 * @internal
 *
 * Set name##-node value in shared memory.
 *
 * @ingroup state_gpu
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @param val Value to set.
 *
 * Sets the value of the name##-node pertaining to the current thread, in the
 * trajectory pertaining to the current thread.
 */
template<class B, class X, int Xo, int Yo, int Zo>
CUDA_FUNC_DEVICE void shared_put(const Coord& cox, const real& val);

/**
 * @internal
 *
 * Facade for state in shared memory.
 *
 * @ingroup state_gpu
 */
struct shared {
  static const bool on_device = true;

  template<class B, class X, int Xo, int Yo, int Zo>
  static CUDA_FUNC_DEVICE real fetch(const int p, const Coord& cox) {
    return shared_fetch<B,X,Xo,Yo,Zo>(cox);
  }

  template<class B, class X, int Xo, int Yo, int Zo>
  static CUDA_FUNC_DEVICE void put(const int p, const Coord& cox,
      const real& val) {
    shared_put<B,X,Xo,Yo,Zo>(val);
  }
};

}

#include "constant.cuh"
#include "global.cuh"

/**
 * @internal
 *
 * @def SHARED_INIT_DEF
 *
 * Macro for shared memory init function definitions.
 */
#define SHARED_INIT_DEF(name, Name) \
  inline void bi::shared_init_##name() { \
    real* share = reinterpret_cast<real*>(shared_mem); \
    const int id = threadIdx.y; \
    const int p = blockIdx.x*blockDim.x + threadIdx.x; \
    const int q = id*blockDim.x + threadIdx.x; \
    \
    if (p < constP) { \
      share[q] = global##Name##State(p, id); \
    } \
  }

SHARED_INIT_DEF(c, C)
SHARED_INIT_DEF(d, D)

/**
 * @internal
 *
 * @def SHARED_COMMIT_DEF
 *
 * Macro for shared memory commit function definitions.
 */
#define SHARED_COMMIT_DEF(name, Name) \
  inline void bi::shared_commit_##name() { \
    real* share = reinterpret_cast<real*>(shared_mem); \
    const int id = threadIdx.y; \
    const int p = blockIdx.x*blockDim.x + threadIdx.x; \
    const int q = id*blockDim.x + threadIdx.x; \
    \
    if (p < constP) { \
      global##Name##State(p, id) = share[q]; \
    } \
  }

SHARED_COMMIT_DEF(c, C)
SHARED_COMMIT_DEF(d, D)

template<class B, class X, int Xo, int Yo, int Zo>
inline real bi::shared_fetch(const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();
  const int q = i*blockDim.x + threadIdx.x;

  return shared_mem[q];
}

template<class B, class X, int Xo, int Yo, int Zo>
inline void bi::shared_put(const Coord& cox, const real& val) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();
  const int q = i*blockDim.x + threadIdx.x;

  shared_mem[q] = val;
}

#endif

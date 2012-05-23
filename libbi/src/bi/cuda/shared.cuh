/**
 * @file
 *
 * Functions for manipulating state objects in shared memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Shared memory is supported for an arbitrary subset of the nodes of a model.
 * A type list is used to determine which nodes can be found in shared
 * memory, with others deferred to global memory.
 */
#ifndef BI_CUDA_SHARED_CUH
#define BI_CUDA_SHARED_CUH

#include "cuda.hpp"

/**
 * Shared memory.
 */
extern CUDA_VAR_SHARED real shared_mem[];

namespace bi {
/**
 * Initialise shared host memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param p Trajectory id.
 * @param i Variable id.
 *
 * Initialise shared host memory by copying the relevant variables of the
 * given trajectory from host memory.
 */
template<class B, class S>
CUDA_FUNC_DEVICE void shared_init(const int p, const int i);

/**
 * Commit shared host memory changes.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param p Trajectory id.
 * @param i Variable id.
 *
 * Write shared host memory changes to the given output.
 */
template<class B, class S>
CUDA_FUNC_DEVICE void shared_commit(const int p, const int i);

/**
 * Fetch node value from shared memory.
 *
 * @ingroup state_gpu
 *
 * @tparam S Type list giving nodes in shared memory.
 * @tparam X Node type.
 *
 * @param ix Serial coordinate.
 *
 * @return Value.
 */
template<class S, class X>
CUDA_FUNC_DEVICE real shared_fetch(const int ix);

/**
 * Set node value in shared memory.
 *
 * @ingroup state_gpu
 *
 * @tparam S Type list giving nodes in shared memory.
 * @tparam X Node type.
 *
 * @param ix Serial coordinate.
 * @param val Value to set.
 */
template<class S, class X>
CUDA_FUNC_DEVICE void shared_put(const int ix, const real& val);

/**
 * Facade for state in shared memory.
 *
 * @ingroup state_gpu
 *
 * @tparam S Type list giving nodes in shared memory.
 */
template<class S>
struct shared {
  static const bool on_device = true;

  template<class B, class X>
  static CUDA_FUNC_DEVICE real fetch(const int p, const int ix) {
    if (contains<S,X>::value) {
      return shared_fetch<S,X>(ix);
    } else {
      return global_fetch<B,X>(p, ix);
    }
  }

  template<class B, class X>
  static CUDA_FUNC_DEVICE void put(const int p, const int ix,
      const real& val) {
    if (contains<S,X>::value) {
      return shared_put<S,X>(ix, val);
    } else {
      return global_put<B,X>(p, ix, val);
    }
  }

  template<class B, class X>
  static CUDA_FUNC_DEVICE void init(const int p, const int ix) {
    shared_put<S,X>(ix, global_fetch<B,X>(p, ix));
  }

  template<class B, class X>
  static CUDA_FUNC_DEVICE void commit(const int p, const int ix) {
    global_put<B,X>(p, ix, shared_fetch<S,X>(ix));
  }
};

}

#include "constant.cuh"
#include "global.cuh"
#include "shared_init_visitor.cuh"
#include "shared_commit_visitor.cuh"

template<class B, class S>
inline void bi::shared_init(const int p, const int i) {
  shared<S> pax;
  shared_init_visitor<B,S>::accept(pax, p, i);
}

template<class B, class S>
inline void bi::shared_commit(const int p, const int i) {
  shared<S> pax;
  shared_commit_visitor<B,S>::accept(pax, p, i);
}

template<class S, class X>
inline real bi::shared_fetch(const int ix) {
  const int i = target_start<S,X>::value + ix;
  const int j = blockDim.x*i + threadIdx.x;

  return shared_mem[j];
}

template<class S, class X>
inline void bi::shared_put(const int ix, const real& val) {
  const int i = target_start<S,X>::value + ix;
  const int j = blockDim.x*i + threadIdx.x;

  shared_mem[j] = val;
}

#endif

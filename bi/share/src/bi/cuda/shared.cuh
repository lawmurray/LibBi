/**
 * @file
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_SHARED_CUH
#define BI_CUDA_SHARED_CUH

#include "global.cuh"

/**
 * Shared memory.
 */
extern CUDA_VAR_SHARED real shared_mem[];

namespace bi {
/**
 * Initialise CUDA shared memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared memory.
 *
 * @param s State.
 * @param p Trajectory id.
 * @param i Variable id.
 *
 * Initialise shared memory by copying the relevant variables of the
 * given trajectory from global memory.
 */
template<class B, class S>
CUDA_FUNC_DEVICE void shared_init(State<B,ON_DEVICE>& s, const int p,
    const int i);

/**
 * Commit CUDA shared memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared memory.
 *
 * @param s[out] State.
 * @param p Trajectory id.
 * @param i Variable id.
 *
 * Write shared memory changes to global memory.
 */
template<class B, class S>
CUDA_FUNC_DEVICE void shared_commit(State<B,ON_DEVICE>& s, const int p,
    const int i);

/**
 * Facade for state in CUDA shared memory.
 *
 * @ingroup state_gpu
 *
 * @tparam S Type list giving nodes in shared memory.
 */
template<class S>
struct shared: public global {
  static const bool on_device = true;

  /**
   * Fetch variable.
   *
   * @ingroup state
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id. This is ignored, the correct offset in shared
   * memory is determined using the thread number.
   *
   * @return Variable.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE vector_reference_type fetch(State<B,ON_DEVICE>& s,
      const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id. This is ignored, the correct offset in shared
   * memory is determined using the thread number.
   *
   * @return Variable.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE vector_reference_type fetch(
      const State<B,ON_DEVICE>& s, const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id. This is ignored, the correct offset in shared
   * memory is determined using the thread number.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE real& fetch(State<B,ON_DEVICE>& s, const int p,
      const int ix);

  /**
   * Fetch variable.
   *
   * @ingroup state
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id. This is ignored, the correct offset in shared
   * memory is determined using the thread number.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE const real& fetch(const State<B,ON_DEVICE>& s,
      const int p, const int ix);

};

}

#include "shared_init_visitor.cuh"
#include "shared_commit_visitor.cuh"
#include "../traits/block_traits.hpp"

template<class B, class S>
inline void bi::shared_init(State<B,ON_DEVICE>& s, const int p, const int i) {
  shared_init_visitor<B,S,S>::accept(s, p, i);
}

template<class B, class S>
inline void bi::shared_commit(State<B,ON_DEVICE>& s, const int p,
    const int i) {
  shared_commit_visitor<B,S,S>::accept(s, p, i);
}

template<class S>
template<class B, class X>
typename bi::shared<S>::vector_reference_type bi::shared<S>::fetch(
    State<B,ON_DEVICE>& s, const int p) {
  if (block_contains_target<S,X>::value) {
    const int start = target_start<S,X>::value;
    const int size = target_size<X>::value;
    const int Q = blockDim.x;
    const int q = threadIdx.x;

    return vector_reference_type(shared_mem + Q * start + q, size, Q);
  } else {
    return global::template fetch<B,X>(s, p);
  }
}

template<class S>
template<class B, class X>
typename bi::shared<S>::vector_reference_type bi::shared<S>::fetch(
    const State<B,ON_DEVICE>& s, const int p) {
  if (block_contains_target<S,X>::value) {
    const int start = target_start<S,X>::value;
    const int size = target_size<X>::value;
    const int Q = blockDim.x;
    const int q = threadIdx.x;

    return vector_reference_type(shared_mem + Q * start + q, size, Q);
  } else {
    return global::template fetch<B,X>(s, p);
  }
}

template<class S>
template<class B, class X>
real& bi::shared<S>::fetch(State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  if (block_contains_target<S,X>::value) {
    const int start = target_start<S,X>::value;
    const int Q = blockDim.x;
    const int q = threadIdx.x;

    return shared_mem[Q * (start + ix) + q];
  } else {
    return global::template fetch<B,X>(s, p, ix);
  }
}

template<class S>
template<class B, class X>
const real& bi::shared<S>::fetch(const State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  if (block_contains_target<S,X>::value) {
    const int start = target_start<S,X>::value;
    const int Q = blockDim.x;
    const int q = threadIdx.x;

    return shared_mem[Q * (start + ix) + q];
  } else {
    return global::template fetch<B,X>(s, p, ix);
  }
}

#endif

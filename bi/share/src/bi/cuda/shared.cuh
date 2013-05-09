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

#include "global_load_visitor.cuh"
#include "global_store_visitor.cuh"
#include "../traits/block_traits.hpp"
#include "../traits/var_traits.hpp"

template<class B, class S>
inline void bi::shared_init(State<B,ON_DEVICE>& s, const int p, const int i) {
  const int Q = blockDim.x;
  const int q = threadIdx.x;

  for (int j = threadIdx.y; j < B::ND; j += blockDim.y) {
    shared_mem[j*Q + q] = s.get(D_VAR)(p, j);
  }
  //global_load_visitor<B,S,S>::accept(s, p, i);
}

template<class B, class S>
inline void bi::shared_commit(State<B,ON_DEVICE>& s, const int p,
    const int i) {
  const int Q = blockDim.x;
  const int q = threadIdx.x;

  for (int j = threadIdx.y; j < B::ND; j += blockDim.y) {
    s.get(D_VAR)(p, j) = shared_mem[j*Q + q];
  }
  //global_store_visitor<B,S,S>::accept(s, p, i);
}

template<class S>
template<class B, class X>
typename bi::shared<S>::vector_reference_type bi::shared<S>::fetch(
    State<B,ON_DEVICE>& s, const int p) {
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;
  const int Q = blockDim.x;
  const int q = threadIdx.x;

  return vector_reference_type(shared_mem + Q * start + q, size, Q);
}

template<class S>
template<class B, class X>
typename bi::shared<S>::vector_reference_type bi::shared<S>::fetch(
    const State<B,ON_DEVICE>& s, const int p) {
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;
  const int Q = blockDim.x;
  const int q = threadIdx.x;

  return vector_reference_type(shared_mem + Q * start + q, size, Q);
}

template<class S>
template<class B, class X>
real& bi::shared<S>::fetch(State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  const int start = var_start<X>::value;
  const int Q = blockDim.x;
  const int q = threadIdx.x;

  return shared_mem[Q * (start + ix) + q];
}

template<class S>
template<class B, class X>
const real& bi::shared<S>::fetch(const State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  const int start = var_start<X>::value;
  const int Q = blockDim.x;
  const int q = threadIdx.x;

  return shared_mem[Q * (start + ix) + q];
}

#endif

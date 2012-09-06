/**
 * @file
 *
 * Functions for manipulating state objects in shared host memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * shared host memory is supported for an arbitrary subset of the nodes of a model.
 * A type list is used to determine which nodes can be found in shared
 * memory, with others deferred to global memory.
 */
#ifndef BI_SSE_SSESHAREDHOST_HPP
#define BI_SSE_SSESHAREDHOST_HPP

#include "sse_host.hpp"
#include "../host/shared_host.hpp"

extern BI_THREAD bi::host_vector_reference<bi::sse_real>* sseSharedHostState;

#ifdef __ICC
#pragma omp threadprivate(sseSharedHostState)
#endif

namespace bi {
/**
 * Initialise SSE shared host memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param s State.
 * @param p First trajectory id.
 *
 * Initialise SSE shared host memory by copying the relevant variables of the
 * given trajectories from host memory.
 */
template<class B, class S>
void sse_shared_host_init(State<B,ON_HOST>& s, const int p);

/**
 * Commit SSE shared host memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param s[out] State.
 * @param p First trajectory id.
 *
 * Write SSE shared host memory changes to the given output.
 */
template<class B, class S>
void sse_shared_host_commit(State<B,ON_HOST>& s, const int p);

/**
 * Facade for state in shared host memory. Shared host memory is allocated
 * on the stack.
 *
 * @ingroup state_host
 *
 * @tparam S Action type list describing variables in shared host memory.
 */
template<class S>
struct sse_shared_host: public sse_host {
  static const bool on_device = true;

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   *
   * @return Variable.
   */
  template<class B, class X>
  static vector_reference_type fetch(State<B,ON_HOST>& s, const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   *
   * @return Variable.
   */
  template<class B, class X>
  static const vector_reference_type fetch(const State<B,ON_HOST>& s,
      const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class B, class X>
  static sse_real& fetch(State<B,ON_HOST>& s, const int p, const int ix);

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class B, class X>
  static const sse_real& fetch(const State<B,ON_HOST>& s, const int p,
      const int ix);
};

}

#include "sse_shared_host_init_visitor.hpp"
#include "sse_shared_host_commit_visitor.hpp"

template<class B, class S>
inline void bi::sse_shared_host_init(State<B,ON_HOST>& s, const int p) {
  sse_shared_host_init_visitor<B,S,S>::accept(s, p);
}

template<class B, class S>
inline void bi::sse_shared_host_commit(State<B,ON_HOST>& s, const int p) {
  sse_shared_host_commit_visitor<B,S,S>::accept(s, p);
}

template<class S>
template<class B, class X>
typename bi::sse_shared_host<S>::vector_reference_type bi::sse_shared_host<S>::fetch(
    State<B,ON_HOST>& s, const int p) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;
    static const int size = target_size<X>::value;

    return subrange(*sseSharedHostState, start, size);
  } else {
    return sse_host::template fetch<B,X>(s, p);
  }
}

template<class S>
template<class B, class X>
const typename bi::sse_shared_host<S>::vector_reference_type bi::sse_shared_host<
    S>::fetch(const State<B,ON_HOST>& s, const int p) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;
    static const int size = target_size<X>::value;

    return subrange(*sseSharedHostState, start, size);
  } else {
    return sse_host::template fetch<B,X>(s, p);
  }
}

template<class S>
template<class B, class X>
bi::sse_real& bi::sse_shared_host<S>::fetch(State<B,ON_HOST>& s,
    const int p, const int ix) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;

    return (*sseSharedHostState)[start + ix];
  } else {
    return sse_host::template fetch<B,X>(s, p, ix);
  }
}

template<class S>
template<class B, class X>
const bi::sse_real& bi::sse_shared_host<S>::fetch(const State<B,ON_HOST>& s,
    const int p, const int ix) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;

    return (*sseSharedHostState)[start + ix];
  } else {
    return sse_host::template fetch<B,X>(s, p, ix);
  }
}

#endif

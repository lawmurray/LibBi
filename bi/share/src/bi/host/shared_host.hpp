/**
 * @file
 *
 * Functions for manipulating state objects in shared host memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * shared host memory is supported for an arbitrary subset of the variables of a model.
 * A type list is used to determine which variables can be found in shared
 * memory, with others deferred to global memory.
 */
#ifndef BI_HOST_SHAREDHOST_HPP
#define BI_HOST_SHAREDHOST_HPP

#include "host.hpp"
#include "../misc/omp.hpp"
#include "../traits/target_traits.hpp"
#include "../traits/block_traits.hpp"

extern BI_THREAD bi::host_vector_reference<real>* sharedHostState;

#ifdef __ICC
#pragma omp threadprivate(sharedHostState)
#endif

namespace bi {
/**
 * Initialise shared host memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param s State.
 * @param p Trajectory id.
 *
 * Initialise shared host memory by copying the relevant variables of the
 * given trajectory from host memory.
 */
template<class B, class S>
void shared_host_init(State<B,ON_HOST>& s, const int p);

/**
 * Commit shared host memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param s[out] State.
 * @param p Trajectory id.
 *
 * Write shared host memory changes to main memory.
 */
template<class B, class S>
void shared_host_commit(State<B,ON_HOST>& s, const int p);

/**
 * Facade for state in shared host memory. Shared host memory is allocated
 * on the stack.
 *
 * @ingroup state_host
 *
 * @tparam S Action type list describing variables in shared host memory.
 */
template<class S>
struct shared_host: public host {
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
  static const vector_reference_type fetch(const State<B,ON_HOST>& s, const int p);

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
  static real& fetch(State<B,ON_HOST>& s, const int p, const int ix);

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
  static const real& fetch(const State<B,ON_HOST>& s, const int p, const int ix);

};

}

#include "shared_host_init_visitor.hpp"
#include "shared_host_commit_visitor.hpp"

template<class B, class S>
inline void bi::shared_host_init(State<B,ON_HOST>& s, const int p) {
  shared_host_init_visitor<B,S,S>::accept(s, p);
}

template<class B, class S>
inline void bi::shared_host_commit(State<B,ON_HOST>& s, const int p) {
  shared_host_commit_visitor<B,S,S>::accept(s, p);
}

template<class S>
template<class B, class X>
typename bi::shared_host<S>::vector_reference_type bi::shared_host<S>::fetch(
    State<B,ON_HOST>& s, const int p) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;
    static const int size = target_size<X>::value;

    return subrange(*sharedHostState, start, size);
  } else {
    return host::template fetch<B,X>(s, p);
  }
}

template<class S>
template<class B, class X>
const typename bi::shared_host<S>::vector_reference_type bi::shared_host<S>::fetch(
    const State<B,ON_HOST>& s, const int p) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;
    static const int size = target_size<X>::value;

    return subrange(*sharedHostState, start, size);
  } else {
    return host::template fetch<B,X>(s, p);
  }
}

template<class S>
template<class B, class X>
real& bi::shared_host<S>::fetch(State<B,ON_HOST>& s, const int p,
    const int ix) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;

    return (*sharedHostState)[start + ix];
  } else {
    return host::template fetch<B,X>(s, p, ix);
  }
}

template<class S>
template<class B, class X>
const real& bi::shared_host<S>::fetch(const State<B,ON_HOST>& s, const int p,
    const int ix) {
  if (block_contains_target<S,X>::value) {
    static const int start = target_start<S,X>::value;

    return (*sharedHostState)[start + ix];
  } else {
    return host::template fetch<B,X>(s, p, ix);
  }
}

#endif

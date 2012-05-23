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
#ifndef BI_HOST_SHAREDHOST_HPP
#define BI_HOST_SHAREDHOST_HPP

#include "host.hpp"
#include "../misc/omp.hpp"
#include "../traits/target_traits.hpp"
#include "../traits/block_traits.hpp"

#include "../misc/assert.hpp"

extern BI_THREAD bi::host_vector<real>* sharedHostState;

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
 * @param p Trajectory id.
 *
 * Initialise shared host memory by copying the relevant variables of the
 * given trajectory from host memory.
 */
template<class B, class S>
void shared_host_init(const int p);

/**
 * Commit shared host memory changes.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param p Trajectory id.
 *
 * Write shared host memory changes to the given output.
 */
template<class B, class S>
void shared_host_commit(const int p);

/**
 * Fetch node value from shared host memory.
 *
 * @ingroup state_host
 *
 * @tparam S Action type list describing nodes in shared host memory.
 * @tparam X Target type.
 *
 * @param ix Serial coordinate.
 *
 * @return Value.
 */
template<class S, class X>
real shared_host_fetch(const int ix);

/**
 * Set node value in shared host memory.
 *
 * @ingroup state_host
 *
 * @tparam S Action type list describing nodes in shared host memory.
 * @tparam X Target type.
 *
 * @param ix Serial coordinate.
 * @param val Value to set.
 */
template<class S, class X>
void shared_host_put(const int ix, const real& val);

/**
 * Facade for state in shared host memory.
 *
 * @ingroup state_host
 *
 * @tparam S Action type list describing nodes in shared host memory.
 */
template<class S>
struct shared_host {
  static const bool on_device = true;

  template<class B, class X>
  static real fetch(const int p, const int ix) {
    if (block_contains_target<S,X>::value) {
      return shared_host_fetch<S,X>(ix);
    } else {
      return host_fetch<B,X>(p, ix);
    }
  }

  template<class B, class X>
  static void put(const int p, const int ix, const real& val) {
    if (block_contains_target<S,X>::value) {
      return shared_host_put<S,X>(ix, val);
    } else {
      return host_put<B,X>(p, ix, val);
    }
  }

  template<class B, class X>
  static void init(const int p, const int ix) {
    assert ((block_contains_target<S,X>::value));
    shared_host_put<S,X>(ix, host_fetch<B,X>(p, ix));
  }

  template<class B, class X>
  static void commit(const int p, const int ix) {
    assert ((block_contains_target<S,X>::value));
    host_put<B,X>(p, ix, shared_host_fetch<S,X>(ix));
  }
};

}

#include "shared_host_init_visitor.hpp"
#include "shared_host_commit_visitor.hpp"

template<class B, class S>
inline void bi::shared_host_init(const int p) {
  delete sharedHostState;
  sharedHostState = new host_vector<real>(block_size<S>::value);

  shared_host<S> pax;
  shared_host_init_visitor<B,S>::accept(pax, p);
}

template<class B, class S>
inline void bi::shared_host_commit(const int p) {
  shared_host<S> pax;
  shared_host_commit_visitor<B,S>::accept(pax, p);
}

template<class S, class X>
inline real bi::shared_host_fetch(const int ix) {
  const int i = target_start<S,X>::value + ix;

  return (*sharedHostState)(i);
}

template<class S, class X>
inline void bi::shared_host_put(const int ix, const real& val) {
  const int i = target_start<S,X>::value + ix;

  (*sharedHostState)(i) = val;
}

#endif

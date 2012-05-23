/**
 * @file
 *
 * Functions for manipulating state objects in shared host memory.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2279 $
 * $Date: 2011-12-13 13:09:45 +0800 (Tue, 13 Dec 2011) $
 *
 * shared host memory is supported for an arbitrary subset of the nodes of a model.
 * A type list is used to determine which nodes can be found in shared
 * memory, with others deferred to global memory.
 */
#ifndef BI_SSE_SSESHAREDHOST_HPP
#define BI_SSE_SSESHAREDHOST_HPP

#include "sse_host.hpp"
#include "sse.hpp"
#include "../misc/omp.hpp"
#include "../traits/target_traits.hpp"
#include "../traits/block_traits.hpp"

extern BI_THREAD bi::host_vector<bi::sse_real>* sseSharedHostState;

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
 * @param p Trajectory id.
 *
 * Initialise shared host memory by copying the relevant variables of the
 * given trajectory from host memory.
 */
template<class B, class S>
void sse_shared_host_init(const int p);

/**
 * Commit SSE shared host memory changes.
 *
 * @tparam B Model type.
 * @tparam S Action type list describing variables in shared host memory.
 *
 * @param p Trajectory id.
 *
 * Write shared host memory changes to the given output.
 */
template<class B, class S>
void sse_shared_host_commit(const int p);

/**
 * Fetch node value from SSE shared host memory.
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
sse_real sse_shared_host_fetch(const int ix);

/**
 * Set node value in SSE shared host memory.
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
void sse_shared_host_put(const int ix, const sse_real& val);

/**
 * Facade for state in SSE shared host memory.
 *
 * @ingroup state_host
 *
 * @tparam S Action type list describing nodes in shared host memory.
 */
template<class S>
struct sse_shared_host {
  static const bool on_device = true;

  template<class B, class X>
  static sse_real fetch(const int p, const int ix) {
    if (block_contains_target<S,X>::value) {
      return sse_shared_host_fetch<S,X>(ix);
    } else {
      return sse_host_fetch<B,X>(p, ix);
    }
  }

  template<class B, class X>
  static void put(const int p, const int ix, const sse_real& val) {
    if (block_contains_target<S,X>::value) {
      return sse_shared_host_put<S,X>(ix, val);
    } else {
      return sse_host_put<B,X>(p, ix, val);
    }
  }

  template<class B, class X>
  static void init(const int p, const int ix) {
    assert ((block_contains_target<S,X>::value));
    sse_shared_host_put<S,X>(ix, sse_host_fetch<B,X>(p, ix));
  }

  template<class B, class X>
  static void commit(const int p, const int ix) {
    assert ((block_contains_target<S,X>::value));
    sse_host_put<B,X>(p, ix, sse_shared_host_fetch<S,X>(ix));
  }
};

}

#include "sse_shared_host_init_visitor.hpp"
#include "sse_shared_host_commit_visitor.hpp"

template<class B, class S>
inline void bi::sse_shared_host_init(const int p) {
  delete sseSharedHostState;
  sseSharedHostState = new host_vector<sse_real>(block_size<S>::value);

  sse_shared_host<S> pax;
  sse_shared_host_init_visitor<B,S>::accept(pax, p);
}

template<class B, class S>
inline void bi::sse_shared_host_commit(const int p) {
  sse_shared_host<S> pax;
  sse_shared_host_commit_visitor<B,S>::accept(pax, p);
}

template<class S, class X>
inline bi::sse_real bi::sse_shared_host_fetch(const int ix) {
  const int i = target_start<S,X>::value + ix;

  return (*sseSharedHostState)(i);
}

template<class S, class X>
inline void bi::sse_shared_host_put(const int ix, const sse_real& val) {
  const int i = target_start<S,X>::value + ix;

  (*sseSharedHostState)(i) = val;
}

#endif

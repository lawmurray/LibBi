/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_SSESHAREDHOSTCOMMITVISITOR_HPP
#define BI_SSE_SSESHAREDHOSTCOMMITVISITOR_HPP

#include "sse_shared_host.hpp"

namespace bi {
/**
 * Visitor for committing SSE shared memory.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list, giving variables in shared host memory.
 * @tparam S2 Action type list.
 */
template<class B, class S1, class S2>
class sse_shared_host_commit_visitor {
public:
  /**
   * Commit SSE shared memory.
   *
   * @param s[out] State.
   * @param p Trajectory id.
   */
  static void accept(State<B,ON_HOST>& s, const int p);
};

/**
 * @internal
 *
 * Base case of sse_shared_host_commit_visitor.
 */
template<class B, class S1>
class sse_shared_host_commit_visitor<B,S1,empty_typelist> {
public:
  static void accept(State<B,ON_HOST>& s, const int p) {
    //
  }
};
}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/target_traits.hpp"
#include "../math/view.hpp"

template<class B, class S1, class S2>
inline void bi::sse_shared_host_commit_visitor<B,S1,S2>::accept(
    State<B,ON_HOST>& s, const int p) {
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::target_type target_type;

  static const int start = target_start<S1,target_type>::value;
  static const int size = target_size<target_type>::value;

  sse_host::fetch<B,target_type>(s, p) = subrange(*sseSharedHostState, start, size);
  sse_shared_host_commit_visitor<B,S1,pop_front>::accept(s, p);
}

#endif

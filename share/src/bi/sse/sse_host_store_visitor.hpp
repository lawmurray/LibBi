/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_SSEHOSTSTOREVISITOR_HPP
#define BI_SSE_SSEHOSTSTOREVISITOR_HPP

namespace bi {
/**
 * Visitor for sse_host_store.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list, giving targets in block.
 * @tparam S2 Action type list.
 */
template<class B, class S1, class S2>
class sse_host_store_visitor {
public:
  /**
   * Accept.
   *
   * @tparam V1 Vector type.
   *
   * @param s[in,out] State.
   * @param p Trajectory id.
   * @param x Vector.
   */
  template<class V1>
  static void accept(State<B,ON_HOST>& s, const int p, const V1 x);
};

/**
 * @internal
 *
 * Base case of sse_host_store_visitor.
 */
template<class B, class S1>
class sse_host_store_visitor<B,S1,empty_typelist> {
public:
  template<class V1>
  static void accept(State<B,ON_HOST>& s, const int p, const V1 x) {
    //
  }
};
}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/action_traits.hpp"
#include "../math/view.hpp"

template<class B, class S1, class S2>
template<class V1>
inline void bi::sse_host_store_visitor<B,S1,S2>::accept(
    State<B,ON_HOST>& s, const int p, const V1 x) {
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename front::coord_type coord_type;

  sse_host::fetch<B,target_type>(s, p) = subrange(x, action_start<S1,front>::value, action_size<front>::value);
  sse_host_store_visitor<B,S1,pop_front>::accept(s, p, x);
}

#endif

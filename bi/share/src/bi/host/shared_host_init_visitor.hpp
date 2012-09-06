/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_SHAREDHOSTINITVISITOR_HPP
#define BI_HOST_SHAREDHOSTINITVISITOR_HPP

#include "shared_host.hpp"

namespace bi {
/**
 * Visitor for initialising shared host memory.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list, giving variables in shared host memory.
 * @tparam S2 Action type list.
 */
template<class B, class S1, class S2>
class shared_host_init_visitor {
public:
  /**
   * Initialise shared host memory.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  static void accept(State<B,ON_HOST>& s, const int p);
};

/**
 * @internal
 *
 * Base case of shared_host_init_visitor.
 */
template<class B, class S1>
class shared_host_init_visitor<B,S1,empty_typelist> {
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
inline void bi::shared_host_init_visitor<B,S1,S2>::accept(
    State<B,ON_HOST>& s, const int p) {
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::target_type target_type;

  shared_host<S1>::template fetch<B,target_type>(s, p) = host::fetch<B,target_type>(s, p);
  shared_host_init_visitor<B,S1,pop_front>::accept(s, p);
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_HOSTLOADVISITOR_HPP
#define BI_HOST_HOSTLOADVISITOR_HPP

namespace bi {
/**
 * Visitor for host_load.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list, giving variables in shared host memory.
 * @tparam S2 Action type list.
 */
template<class B, class S1, class S2>
class host_load_visitor {
public:
  /**
   * Accept.
   *
   * @tparam V1 Vector type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param[out] x Vector.
   */
  template<class V1>
  static void accept(State<B,ON_HOST>& s, const int p, V1 x);
};

/**
 * @internal
 *
 * Base case of host_load_visitor.
 */
template<class B, class S1>
class host_load_visitor<B,S1,empty_typelist> {
public:
  template<class V1>
  static void accept(State<B,ON_HOST>& s, const int p, V1 x) {
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
inline void bi::host_load_visitor<B,S1,S2>::accept(
    State<B,ON_HOST>& s, const int p, V1 x) {
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename front::coord_type coord_type;

  int ix = 0;
  coord_type cox;
  while (ix < action_size<front>::value) {
    x(action_start<S1,front>::value + ix) = host::fetch<B,target_type>(s, p)(cox.index());
    ++cox;
    ++ix;
  }
  host_load_visitor<B,S1,pop_front>::accept(s, p, x);
}

#endif

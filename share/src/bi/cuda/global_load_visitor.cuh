/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_GLOBALLOADVISITOR_HPP
#define BI_CUDA_GLOBALLOADVISITOR_HPP

namespace bi {
/**
 * Visitor for initialising shared memory.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list, giving variables in shared memory.
 * @tparam S2 Action type list.
 */
template<class B, class S1, class S2>
class global_load_visitor {
public:
  /**
   * Initialise shared memory.
   *
   * @param[out] s State.
   * @param p Trajectory id.
   * @param i Variable id.
   */
  static CUDA_FUNC_DEVICE real& accept(State<B,ON_DEVICE>& s,
      const int p, const int i);
};

/**
 * @internal
 *
 * Base case of global_load_visitor.
 */
template<class B, class S1>
class global_load_visitor<B,S1,empty_typelist> {
public:
  static CUDA_FUNC_DEVICE real& accept(State<B,ON_DEVICE>& s,
      const int p, const int i) {
    //
  }
};
}

#include "shared.cuh"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/action_traits.hpp"

template<class B, class S1, class S2>
inline real& bi::global_load_visitor<B,S1,S2>::accept(State<B,ON_DEVICE>& s,
    const int p, const int i) {
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename front::coord_type coord_type;

  const int size = action_size<front>::value;

  if (i < size) {
    coord_type cox(i);
    return global::template fetch<B,target_type>(s, p, cox.index());
  } else {
    return global_load_visitor<B,S1,pop_front>::accept(s, p, i - size);
  }
}

#endif

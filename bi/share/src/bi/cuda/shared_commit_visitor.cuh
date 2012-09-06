/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_SHAREDCOMMITVISITOR_HPP
#define BI_CUDA_SHAREDCOMMITVISITOR_HPP

namespace bi {
/**
 * Visitor for committing shared memory.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list, giving variables in shared memory.
 * @tparam S2 Action type list.
 */
template<class B, class S1, class S2>
class shared_commit_visitor {
public:
  /**
   * Commit shared memory.
   *
   * @param[out] s State.
   * @param p Trajectory id.
   * @param i Variable id.
   */
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const int p,
      const int i);
};

/**
 * @internal
 *
 * Base case of shared_commit_visitor.
 */
template<class B, class S1>
class shared_commit_visitor<B,S1,empty_typelist> {
public:
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const int p,
      const int i) {
    //
  }
};
}

#include "shared.cuh"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/target_traits.hpp"

template<class B, class S1, class S2>
inline void bi::shared_commit_visitor<B,S1,S2>::accept(State<B,ON_DEVICE>& s,
    const int p, const int i) {
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::target_type target_type;

  const int size = target_size<target_type>::value;

  if (i < size) {
    global::fetch<B,target_type>(s, p, i) =
        shared<S1>::template fetch<B,target_type>(s, p, i);
  } else {
    shared_commit_visitor<B,S1,pop_front>::accept(s, p, i - size);
  }
}

#endif

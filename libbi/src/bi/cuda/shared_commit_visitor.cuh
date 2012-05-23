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
 * @tparam S Action type list, giving variables in shared memory.
 */
template<class B, class S>
class shared_commit_visitor {
public:
  /**
   * Initialise shared memory.
   *
   * @param p Trajectory id.
   * @param i Variable id.
   */
  template<class PX>
  static CUDA_FUNC_DEVICE void accept(PX& pax, const int p, const int i);
};

/**
 * @internal
 *
 * Base case of shared_commit_visitor.
 */
template<class B>
class shared_commit_visitor<B,empty_typelist> {
public:
  template<class PX>
  static CUDA_FUNC_DEVICE void accept(PX& pax, const int p, const int i) {
    //
  }
};
}

#include "shared.cuh"
#include "global.cuh"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/target_traits.hpp"

template<class B, class S>
template<class PX>
inline void bi::shared_commit_visitor<B,S>::accept(PX& pax, const int p,
    const int i) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename target_type::coord_type coord_type;
  const int size = target_size<target_type>::value;

  if (i < size) {
    pax.template commit<B,target_type>(p, i);
  } else {
    shared_commit_visitor<B,pop_front>::accept(pax, p, i - size);
  }
}


#endif

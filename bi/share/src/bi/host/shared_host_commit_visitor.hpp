/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_SHAREDHOSTCOMMITVISITOR_HPP
#define BI_HOST_SHAREDHOSTCOMMITVISITOR_HPP

namespace bi {
/**
 * Visitor for committing shared memory.
 *
 * @tparam B Model type.
 * @tparam S Action type list, giving variables in shared host memory.
 */
template<class B, class S>
class shared_host_commit_visitor {
public:
  /**
   * Initialise shared memory.
   *
   * @param p Trajectory id.
   */
  template<class PX>
  static void accept(PX& pax, const int p);
};

/**
 * @internal
 *
 * Base case of shared_host_commit_visitor.
 */
template<class B>
class shared_host_commit_visitor<B,empty_typelist> {
public:
  template<class PX>
  static void accept(PX& pax, const int p) {
    //
  }
};
}

#include "shared_host.hpp"
#include "host.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/target_traits.hpp"

template<class B, class S>
template<class PX>
inline void bi::shared_host_commit_visitor<B,S>::accept(PX& pax, const int p) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename target_type::coord_type coord_type;
  static const int size = target_size<target_type>::value;

  ///@todo Do as vector copy.
  int i;
  for (i = 0; i < size; ++i) {
    pax.template commit<B,target_type>(p, i);
  }
  shared_host_commit_visitor<B,pop_front>::accept(pax, p);
}

#endif

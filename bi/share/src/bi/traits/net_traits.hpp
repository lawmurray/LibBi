/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_NET_TRAITS_HPP
#define BI_TRAITS_NET_TRAITS_HPP

#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../typelist/size.hpp"

namespace bi {
/**
 * Number of nodes in net.
 *
 * @ingroup model_low
 *
 * @tparam S Type list.
 */
template<class S>
struct net_count {
  static const int value = size<S>::value;
};

/**
 * Size of net given by type list.
 *
 * @ingroup model_low
 *
 * @tparam S Type list.
 */
template<class S>
struct net_size {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  static const int value = var_size<front>::value +
      net_size<pop_front>::value;
};

/**
 * @internal
 *
 * Base case of net_size.
 *
 * @ingroup model_low
 */
template<>
struct net_size<empty_typelist> {
  static const int value = 0;
};
}

#endif

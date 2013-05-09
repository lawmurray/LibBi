/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_TARGET_TRAITS_HPP
#define BI_TRAITS_TARGET_TRAITS_HPP

#include "action_traits.hpp"
#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../typelist/size.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Start of target type in action type list (cumulative sum of the sizes of
 * all preceding targets).
 *
 * @ingroup model_low
 *
 * @tparam S Type list.
 * @tparam X Node type.
 */
template<class S, class X>
struct target_start {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;

  static const int value = (equals<target_type,X>::value ? 0 : action_size<front>::value + target_start<pop_front,X>::value);
};

/**
 * @internal
 *
 * @ingroup model_low
 */
template<class X>
struct target_start<empty_typelist,X> {
  static const int value = 0;
};

/**
 * End of target type in action type list (cumulative sum of the sizes of
 * self and all preceding targets).
 *
 * @ingroup model_low
 *
 * @tparam S Action type list.
 * @tparam X Target type.
 */
template<class S, class X>
struct target_end {
  typedef typename front<S>::type front;

  static const int value = target_start<S,X>::value + action_size<front>::value;
};

}

#endif

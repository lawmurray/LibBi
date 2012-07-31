/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_TARGET_TRAITS_HPP
#define BI_TRAITS_TARGET_TRAITS_HPP

#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../typelist/size.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Size of target type.
 *
 * @ingroup model_low
 *
 * @tparam X Target type.
 */
template<class X>
struct target_size {
  static const int value = X::SIZE;
};

/**
 * Number of dimensions associated with target type.
 *
 * @ingroup model_low
 *
 * @tparam X Target type.
 */
template<class X>
struct target_num_dims {
  static const int value = X::NUM_DIMS;
};

/**
 * Index of target in action type list.
 *
 * @ingroup model_low
 *
 * @tparam S Action type list.
 * @tparam X Target type.
 */
template<class S, class X>
struct target_index {
  typedef typename front<S>::type::target_type front;
  typedef typename pop_front<S>::type pop_front;

  /**
   * Starting index of @p X in @p S.
   */
  static const int value = 1 + target_index<pop_front,X>::value;
};

/**
 * @internal
 *
 * Base case of target_index.
 *
 * @ingroup model_low
 */
template<class S>
struct target_index<S,typename front<S>::type::target_type> {
  static const int value = 0;
};

/**
 * @internal
 *
 * Error case of target_index.
 *
 * @ingroup model_low
 */
template<class X>
struct target_index<empty_typelist,X> {
  //
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
  typedef typename front<S>::type::target_type front;
  typedef typename pop_front<S>::type pop_front;

  static const int value = target_size<front>::value +
      (equals<front,X>::value ? 0 : target_end<pop_front,X>::value);
};

/**
 * @internal
 *
 * Error case of target_start.
 *
 * @ingroup model_low
 */
template<class X>
struct target_end<empty_typelist,X> {
  static const int value = 0;
};

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
  static const int value = target_end<S,X>::value - target_size<X>::value;
};

}

#endif

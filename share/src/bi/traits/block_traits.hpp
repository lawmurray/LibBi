/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_BLOCK_TRAITS_HPP
#define BI_TRAITS_BLOCK_TRAITS_HPP

#include "action_traits.hpp"
#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../typelist/size.hpp"
#include "../typelist/contains.hpp"
#include "../typelist/equals.hpp"

namespace bi {
/**
 * Number of actions in block.
 *
 * @ingroup model_low
 *
 * @tparam S Action type list.
 */
template<class S>
struct block_count {
  static const int value = size<S>::value;
};

/**
 * Size of block.
 *
 * @ingroup model_low
 *
 * @tparam S Action type list.
 */
template<class S>
struct block_size {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target;

  static const int value = var_size<target>::value +
      block_size<pop_front>::value;
};

/**
 * @internal
 *
 * Base case of block_size.
 *
 * @ingroup model_low
 */
template<>
struct block_size<empty_typelist> {
  static const int value = 0;
};

/**
 * Does block contain a particular action?
 *
 * @ingroup model_low
 *
 * @tparam S Action type list.
 * @tparam A Action type.
 */
template<class S, class X>
struct block_contains_action {
  static const bool value = contains<S,X>::value;
};

/**
 * Does block contain an action for a particular target?
 *
 * @ingroup model_low
 *
 * @tparam S Action type list.
 * @tparam X Target type.
 */
template<class S, class X>
struct block_contains_target {
  typedef typename front<S>::type::target_type front;
  typedef typename pop_front<S>::type pop_front;

  static const bool value = equals<front,X>::value || block_contains_target<pop_front,X>::value;
};

/**
 * @internal
 *
 * Base case of block_contains_target.
 *
 * @ingroup model_low
 */
template<class X>
struct block_contains_target<empty_typelist,X> {
  static const bool value = false;
};

/**
 * Is this a matrix block?
 */
template<class S>
struct block_is_matrix {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  static const bool value = action_is_matrix<front>::value && block_is_matrix<pop_front>::value;
};

/**
 * @internal
 *
 * Base case of block_is_matrix.
 *
 * @ingroup model_low
 */
template<>
struct block_is_matrix<empty_typelist> {
  static const bool value = true;
};

}

#endif

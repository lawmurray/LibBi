/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_ACTION_TRAITS_HPP
#define BI_TRAITS_ACTION_TRAITS_HPP

namespace bi {
/**
 * Size of action.
 *
 * @ingroup model_low
 *
 * @tparam A Action type.
 */
template<class A>
struct action_size {
  static const int value = A::SIZE;
};

/**
 * Is action a matrix action?
 *
 * @ingroup model_low
 *
 * @tparam A Action type.
 */
template<class A>
struct action_is_matrix {
  static const int value = A::IS_MATRIX;
};

/**
 * Start of action in action type list (cumulative sum of the sizes of
 * all preceding actions).
 *
 * @ingroup model_low
 *
 * @tparam S Type list.
 * @tparam A Action type.
 */
template<class S, class A>
struct action_start {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  static const int value = (equals<front,A>::value ? 0 : action_size<front>::value + action_start<pop_front,A>::value);
};

/**
 * @internal
 *
 * @ingroup model_low
 */
template<class A>
struct action_start<empty_typelist,A> {
  static const int value = 0;
};

/**
 * End of action in action type list (cumulative sum of the sizes of
 * self and all preceding actions).
 *
 * @ingroup model_low
 *
 * @tparam S Action type list.
 * @tparam A Action type.
 */
template<class S, class A>
struct action_end {
  static const int value = action_start<S,A>::value + action_size<A>::value;
};
}

#endif

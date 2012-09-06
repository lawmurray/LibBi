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
}

#endif

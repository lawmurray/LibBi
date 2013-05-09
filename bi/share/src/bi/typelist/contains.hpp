/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_CONTAINS_HPP
#define BI_TYPELIST_CONTAINS_HPP

#include "front.hpp"
#include "pop_front.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation.
 */
template<class tail, class head, class X, unsigned reps>
struct contains_impl {
  static const int value = contains_impl<typename pop_front<tail>::type,typename front<tail>::type,X,reps+1>::value;
};

/**
 * Does type list contain type?
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 * @ptaram X A type.
 */
template<class T, class X>
struct contains {
  static const int value = contains_impl<typename pop_front<T>::type,typename front<T>::type,X,0>::value;
};

/**
 * @internal
 *
 * Implementation, base case.
 */
template<class tail, class X, unsigned reps>
struct contains_impl<tail,X,X,reps> {
  static const bool value = true;
};

/**
 * @internal
 *
 * Implementation, base case.
 */
template<class head, class X, unsigned reps>
struct contains_impl<empty_typelist,head,X,reps> {
  static const bool value = false;
};

}

#endif

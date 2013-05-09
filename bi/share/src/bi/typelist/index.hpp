/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_INDEX_HPP
#define BI_TYPELIST_INDEX_HPP

#include "front.hpp"
#include "pop_front.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation.
 */
template<class tail, class head, class X, unsigned reps>
struct index_impl {
  static const int value = index_impl<typename pop_front<tail>::type,typename front<tail>::type,X,reps+1>::value;
};

/**
 * Index of first occurrence of type in type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 * @ptaram X A type.
 */
template<class T, class X>
struct index {
  static const int value = index_impl<typename pop_front<T>::type,typename front<T>::type,X,0>::value;
};

/**
 * @internal
 *
 * Implementation, base case.
 */
template<class tail, class X, unsigned reps>
struct index_impl<tail,X,X,reps> {
  static const int value = reps;
};

/**
 * @internal
 *
 * Implementation, error case.
 */
template<class head, class X, unsigned reps>
struct index_impl<empty_typelist,head,X,reps> {
  //
};

}

#endif

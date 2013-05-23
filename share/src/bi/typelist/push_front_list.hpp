/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_PUSH_FRONT_LIST_HPP
#define BI_TYPELIST_PUSH_FRONT_LIST_HPP

#include "typelist.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation, front item is different.
 */
template<typelist_marker marker, int reps, class item, class tail,
    class X, int N>
struct push_front_list_impl {
  typedef typelist<TYPELIST_COMPOUND,N,X,typelist<marker,reps,item,tail> > type;
};

/**
 * Push value onto front of a type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 * @tparam X A scalar type.
 * @tparam N Repetitions.
 */
template<class T, class X, int N = 1>
struct push_front_list {
  typedef typename push_front_list_impl<T::marker,T::reps,typename T::item,
      typename T::tail,X,N>::type type;
};

/**
 * @internal
 *
 * Implementation, front item is same.
 */
template<typelist_marker marker, int reps, class tail, class X,
    int N>
struct push_front_list_impl<marker,reps,X,tail,X,N> {
  typedef typelist<marker,reps+N,X,tail> type;
};

/**
 * @internal
 *
 * Special case for empty_typelist.
 */
template<class X, int N>
struct push_front_list<empty_typelist,X,N> {
  typedef typelist<TYPELIST_COMPOUND,N,X,empty_typelist> type;
};
}

#endif

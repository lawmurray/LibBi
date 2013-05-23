/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_PUSH_BACK_HPP
#define BI_TYPELIST_PUSH_BACK_HPP

#include "typelist.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation, recursive.
 */
template<typelist_marker marker, int reps, class item, class tail, class X, int N>
struct push_back_impl {
  typedef typelist<marker,reps,item,typename push_back_impl<tail::marker,tail::reps,typename tail::item,typename tail::tail,X,N>::type> type;
};

/**
 * Push value onto back of a type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 * @tparam X A scalar type.
 * @tparam N Repetitions.
 */
template<class T, class X, int N = 1>
struct push_back {
  typedef typename push_back_impl<T::marker,T::reps,typename T::item,typename T::tail,X,N>::type type;
};

/**
 * @internal
 *
 * Implementation, back item is same.
 */
template<typelist_marker marker, int reps, class X, int N>
struct push_back_impl<marker,reps,X,empty_typelist,X,N> {
  typedef typelist<marker,reps+N,X,empty_typelist> type;
};

/**
 * @internal
 *
 * Implementation, back item is different.
 */
template<typelist_marker marker, int reps, class item, class X, int N>
struct push_back_impl<marker,reps,item,empty_typelist,X,N> {
  typedef typelist<marker,reps,item,typelist<TYPELIST_SCALAR,N,X,empty_typelist> > type;
};

/**
 * @internal
 *
 * Special case for empty_typelist.
 */
template<class X, int N>
struct push_back<empty_typelist,X,N> {
  typedef typelist<TYPELIST_SCALAR,N,X,empty_typelist> type;
};
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_POP_BACK_HPP
#define BI_TYPELIST_POP_BACK_HPP

#include "typelist.hpp"

//#include "boost/static_assert.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation, recursive.
 */
template<typelist_marker marker, int reps, class item, class tail>
struct pop_back_impl {
  typedef typelist<marker,reps,item,typename pop_back_impl<tail::marker,tail::reps,typename tail::item,typename tail::tail>::type> type;
};

/**
 * Remove the back item of a type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct pop_back {
  //BOOST_STATIC_ASSERT(!empty<T>::value);
  typedef typename pop_back_impl<T::marker,T::reps,typename T::item,typename T::tail>::type type;
};

/**
 * @internal
 *
 * Implementation, back item is scalar type with 1 repeat.
 */
template<class item>
struct pop_back_impl<TYPELIST_SCALAR, 1, item, empty_typelist> {
  typedef empty_typelist type;
};

/**
 * @internal
 *
 * Implementation, back item is scalar type with more than 1 repeat.
 */
template<int reps, class item>
struct pop_back_impl<TYPELIST_SCALAR, reps, item, empty_typelist> {
  typedef typelist<TYPELIST_SCALAR, reps-1, item, empty_typelist> type;
};

/**
 * @internal
 *
 * Implementation, back item is list type with 1 repeat.
 */
template<class item>
struct pop_back_impl<TYPELIST_COMPOUND, 1, item, empty_typelist> {
  typedef typename pop_back_impl<item::marker,item::reps,typename item::item,typename item::tail>::type type;
};

/**
 * @internal
 *
 * Implementation, back item is list type with more than 1 repeat.
 */
template<int reps, class item>
struct pop_back_impl<TYPELIST_COMPOUND, reps, item, empty_typelist> {
  typedef typelist<TYPELIST_COMPOUND,reps-1,item,typename pop_back_impl<item::marker,item::reps,typename item::item,typename item::tail>::type> type;
};

}

#endif

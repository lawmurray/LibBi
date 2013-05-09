/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_POP_FRONT_HPP
#define BI_TYPELIST_POP_FRONT_HPP

#include "typelist.hpp"
#include "append.hpp"

//#include "boost/static_assert.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation.
 */
template<typelist_marker marker, int reps, class item, class tail>
struct pop_front_impl {

};

/**
 * Remove the front item of a type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct pop_front {
  //BOOST_STATIC_ASSERT(!empty<T>::value);
  typedef typename pop_front_impl<T::marker,T::reps,typename T::item,typename T::tail>::type type;
};

/**
 * @internal
 *
 * Implementation, front item is scalar type with 1 repeat.
 */
template<class item, class tail>
struct pop_front_impl<TYPELIST_SCALAR, 1, item, tail> {
  typedef tail type;
};

/**
 * @internal
 *
 * Implementation, front item is scalar type with more than 1 repeat.
 */
template<int reps, class item, class tail>
struct pop_front_impl<TYPELIST_SCALAR, reps, item, tail> {
  typedef typelist<TYPELIST_SCALAR, reps-1, item, tail> type;
};

/**
 * @internal
 *
 * Implementation, front item is list type with 1 repeat.
 */
template<class item, class tail>
struct pop_front_impl<TYPELIST_COMPOUND, 1, item, tail> {
  typedef typename append<typename pop_front_impl<item::marker,item::reps,typename item::item,typename item::tail>::type,tail>::type type;
};

/**
 * @internal
 *
 * Implementation, front item is list type with more than 1 repeat.
 */
template<int reps, class item, class tail>
struct pop_front_impl<TYPELIST_COMPOUND, reps, item, tail> {
  typedef typename append<typename pop_front_impl<item::marker,item::reps,typename item::item,typename item::tail>::type, typelist<TYPELIST_COMPOUND,reps-1,item,tail> >::type type;
};

}

#endif

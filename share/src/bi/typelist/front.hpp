/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_FRONT_HPP
#define BI_TYPELIST_FRONT_HPP

#include "typelist.hpp"
#include "empty.hpp"

//#include "boost/static_assert.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation.
 */
template<typelist_marker marker, class item>
struct front_impl {

};

/**
 * Get the first item of a type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct front {
  //BOOST_STATIC_ASSERT(!empty<T>::value);
  typedef typename front_impl<T::marker,typename T::item>::type type;
};

/**
 * @internal
 *
 * Implementation, scalar type at front.
 */
template<class item>
struct front_impl<TYPELIST_SCALAR, item> {
  typedef item type;
};

/**
 * @internal
 *
 * Implementation, list type at front.
 */
template<class T>
struct front_impl<TYPELIST_COMPOUND, T> {
  typedef typename front_impl<T::marker,typename T::item>::type type;
};

}

#endif

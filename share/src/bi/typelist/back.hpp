/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_BACK_HPP
#define BI_TYPELIST_BACK_HPP

#include "typelist.hpp"
#include "empty.hpp"

//#include "boost/static_assert.hpp"

namespace bi {
/**
 * @internal
 *
 * Implementation, recursive.
 */
template<typelist_marker marker, class item, class tail>
struct back_impl {
  typedef typename back_impl<tail::marker, typename tail::item,
      typename tail::tail>::type type;
};

/**
 * Get the last item of a type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct back {
  //BOOST_STATIC_ASSERT(!empty<T>::value);
  typedef typename back_impl<T::marker, typename T::item,
      typename T::tail>::type type;
};

/**
 * @internal
 *
 * Implementation, back item is value.
 */
template<class item>
struct back_impl<TYPELIST_SCALAR,item,empty_typelist> {
  typedef item type;
};

/**
 * @internal
 *
 * Implementation, back item is list.
 */
template<class item>
struct back_impl<TYPELIST_COMPOUND,item,empty_typelist> {
  typedef typename back_impl<item::marker, typename item::item,
      typename item::tail>::type type;
};

}

#endif

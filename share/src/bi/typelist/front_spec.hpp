/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_FRONT_SPEC_HPP
#define BI_TYPELIST_FRONT_SPEC_HPP

#include "typelist.hpp"
#include "empty.hpp"

//#include "boost/static_assert.hpp"

namespace bi {
/**
 * Get front spec of a type list. Used by append.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct front_spec {
  //BOOST_STATIC_ASSERT(!empty<T>::value);
  typedef typelist<T::marker,T::reps,typename T::item,empty_typelist> type;
};

}

#endif

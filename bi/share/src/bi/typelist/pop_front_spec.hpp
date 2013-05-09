/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_POP_FRONT_SPEC_HPP
#define BI_TYPELIST_POP_FRONT_SPEC_HPP

#include "empty.hpp"

//#include "boost/static_assert.hpp"

namespace bi {
/**
 * Pop first spec of a type list. Used by append.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct pop_front_spec {
  //BOOST_STATIC_ASSERT(!empty<T>::value);
  typedef typename T::tail type;
};
}

#endif

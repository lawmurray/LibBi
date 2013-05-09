/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_APPEND_HPP
#define BI_TYPELIST_APPEND_HPP

#include "typelist.hpp"
#include "front_spec.hpp"
#include "push_back_spec.hpp"
#include "pop_front_spec.hpp"

namespace bi {
/**
 * Append two type lists.
 *
 * @ingroup typelist
 *
 * @tparam T1 A type list.
 * @tparam T2 A type list.
 */
template<class T1, class T2>
struct append {
  typedef typename append<typename push_back_spec<T1,typename front_spec<T2>::type>::type,typename pop_front_spec<T2>::type>::type type;
};

/**
 * @internal
 */
template<class T1>
struct append<T1,empty_typelist> {
  typedef T1 type;
};

/**
 * @internal
 */
template<class T2>
struct append<empty_typelist,T2> {
  typedef T2 type;
};

}

#endif

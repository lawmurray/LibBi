/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_PUSH_BACK_SPEC_HPP
#define BI_TYPELIST_PUSH_BACK_SPEC_HPP

#include "typelist.hpp"

namespace bi {
/**
 * Push spec onto back of a type list. Used by append.
 *
 * @ingroup typelist
 *
 * @tparam T1 A type list.
 * @tparam T2 A type list.
 */
template<class T1, class T2>
struct push_back_spec {
  typedef typelist<T1::marker,T1::reps,typename T1::item,typename push_back_spec<typename T1::tail,T2>::type> type;
};

/**
 * @internal
 *
 * Base case.
 */
template<class T2>
struct push_back_spec<empty_typelist,T2> {
  typedef typelist<T2::marker,T2::reps,typename T2::item,empty_typelist> type;
};
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_EMPTY_HPP
#define BI_TYPELIST_EMPTY_HPP

#include "typelist.hpp"

namespace bi {
/**
 * Is type list empty?
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct empty {
  static const bool value = false;
};

/**
 * @internal
 */
template<>
struct empty<empty_typelist> {
  static const bool value = true;
};

}

#endif

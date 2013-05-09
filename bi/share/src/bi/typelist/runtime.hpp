/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_RUNTIME_HPP
#define BI_TYPELIST_RUNTIME_HPP

#include "size.hpp"
#include "../misc/assert.hpp"

#include <typeinfo>

namespace bi {
/**
 * Runtime functions for type lists.
 *
 * @ingroup typelist
 *
 * @tparam TS Type list.
 */
template<class TS>
struct runtime {
  /**
   * Runtime type check of object.
   *
   * @tparam T Object type.
   *
   * @param o Object to check.
   * @param i Index in the type list against which to check.
   *
   * @return True if the type of @p o is the same as the @p i th element of
   * the type list, false otherwise.
   */
  template<class T>
  static bool check(const T& o, const int i) {
    /* pre-conditions */
    BI_ASSERT(i < size<TS>::value);

    typedef typename front<TS>::type front;
    typedef typename pop_front<TS>::type pop_front;

    if (i == 0) {
      return typeid(o) == typeid(front);
    } else {
      return runtime<pop_front>::check(o, i - 1);
    }
  }
};

/**
 * @internal
 *
 * Base case of runtime.
 */
template<>
struct runtime<empty_typelist> {
  template<class T>
  static bool check(const T& o, const int i) {
    BI_ASSERT(false);

    return false;
  }
};

}

#endif

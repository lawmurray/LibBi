/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PRIMITIVE_POINTERLESSEQUAL_HPP
#define BI_PRIMITIVE_POINTERLESSEQUAL_HPP

namespace bi {
/**
 * Comparison of two pointer types. Dereferences the pointers and compares
 * their values with the less-than-or-equal operator.
 */
template<class T>
struct pointer_less_equal {
  bool operator()(const T o1, const T o2) {
    return *o1 <= *o2;
  }
};
}

#endif

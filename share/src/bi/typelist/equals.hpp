/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_EQUALS_HPP
#define BI_TYPELIST_EQUALS_HPP

namespace bi {
/**
 * Type equality check.
 *
 * @ingroup typelist
 *
 * @tparam X A type.
 * @ptaram Y A type.
 */
template<class X, class Y>
struct equals {
  static const bool value = false;
};

/**
 * @internal
 */
template<class X>
struct equals<X,X> {
  static const bool value = true;
};

}

#endif

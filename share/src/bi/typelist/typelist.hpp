/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_TYPELIST_HPP
#define BI_TYPELIST_TYPELIST_HPP

namespace bi {
/**
 * Categories of types in type list.
 *
 * @ingroup typelist
 */
enum typelist_marker {
  /**
   * Scalar type.
   */
  TYPELIST_SCALAR,

  /**
   * List type.
   */
  TYPELIST_COMPOUND
};

/**
 * Empty type list.
 *
 * @ingroup typelist
 */
struct empty_typelist {
  //
};

/**
 * Type list.
 *
 * @ingroup typelist
 *
 * @tparam M Marker.
 * @tparam N Repetitions.
 * @tparam X Item type.
 * @tparam T Tail type.
 */
template<typelist_marker M, int N, class X, class T>
struct typelist {
  /**
   * @internal
   *
   * Category of first type in type list.
   */
  static const typelist_marker marker = M;

  /**
   * @internal
   *
   * Number of times first type in type list is repeated.
   */
  static const int reps = N;

  /**
   * @internal
   *
   * First type in type list.
   */
  typedef X item;

  /**
   * @internal
   *
   * Remainder of type list.
   */
  typedef T tail;

};
}

#endif

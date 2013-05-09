/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_SIZE_HPP
#define BI_TYPELIST_SIZE_HPP

namespace bi {
/**
 * @internal
 *
 * Implementation.
 */
template<typelist_marker marker, int reps, class item, class tail>
struct size_impl {

};

/**
 * Number of nodes in type list.
 *
 * @ingroup typelist
 *
 * @tparam T A type list.
 */
template<class T>
struct size {
  static const int value = size_impl<T::marker,T::reps,typename T::item,typename T::tail>::value;
};

/**
 * @internal
 *
 * Implementation, scalar type, recursive.
 */
template<int reps, class item, class tail>
struct size_impl<TYPELIST_SCALAR,reps,item,tail> {
  static const int value = reps + size_impl<tail::marker,tail::reps,typename tail::item,typename tail::tail>::value;
};

/**
 * @internal
 *
 * Implementation, list type, recursive.
 */
template<int reps, class item, class tail>
struct size_impl<TYPELIST_COMPOUND,reps,item,tail> {
  static const int value = reps*size_impl<item::marker,item::reps,typename item::item,typename item::tail>::value + size_impl<tail::marker,tail::reps,typename tail::item,typename tail::tail>::value;
};

/**
 * @internal
 *
 * Implementation, scalar type, base.
 */
template<int reps, class item>
struct size_impl<TYPELIST_SCALAR,reps,item,empty_typelist> {
  static const int value = reps;
};

/**
 * @internal
 *
 * Implementation, list type, base.
 */
template<int reps, class item>
struct size_impl<TYPELIST_COMPOUND,reps,item,empty_typelist> {
  static const int value = reps*size_impl<item::marker,item::reps,typename item::item,typename item::tail>::value;
};

/**
 * @internal
 *
 * Special case for empty_typelist.
 */
template<>
struct size<empty_typelist> {
  static const int value = 0;
};
}

#endif

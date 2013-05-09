/**
 * @file
 *
 * Macros for building type lists.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_MACROTYPETREE_HPP
#define BI_TYPELIST_MACROTYPETREE_HPP

/**
 * @def BEGIN_TYPETREE(name)
 *
 * Open type tree.
 *
 * @ingroup typelist
 *
 * @arg @c name Name to assign to the type tree.
 */
#define BEGIN_TYPETREE(name) struct name { \
  typedef

/**
 * @def LEAD_NODE(reps, single, ...)
 *
 * Position leaf node in currently open type tree.
 *
 * @ingroup typelist
 *
 * @arg @c reps Number of times to repeat this type.
 * @arg @c single Single type.
 * @arg Remaining variadic arguments are to accommodate template types
 * containing commas.
 */
#define LEAF_NODE(reps, single, ...) bi::typelist<bi::TYPELIST_SCALAR,reps,single,##__VA_ARGS__,bi::empty_typelist>

/**
 * @def INTERNAL_NODE(reps, left, right)
 *
 * Combine two nodes into parent node.
 *
 * @ingroup typelist
 *
 * @arg @c reps Number of times to repeat this type.
 * @arg @c left Left child.
 * @arg @c right Right child.
 */
#define BEGIN_NODE(reps) bi::typelist<bi::TYPELIST_COMPOUND,reps,
#define JOIN_NODE ,
#define END_NODE >
#define NULL_NODE bi::empty_typelist

/**
 * @def END_TYPETREE()
 *
 * Close type tree.
 *
 * @ingroup typelist
 */
#define END_TYPETREE() type; \
};

/**
 * @def GET_TYPETREE(name)
 *
 * Retrieve previously defined type tree.
 *
 * @ingroup typelist
 *
 * @arg @c name Name of the type tree.
 */
#define GET_TYPETREE(name) name::type


#endif

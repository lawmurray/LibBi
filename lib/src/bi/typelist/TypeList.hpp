/**
 * @file
 *
 * Macros for building type lists.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_TYPELIST_HPP
#define BI_MODEL_TYPELIST_HPP

#include "easy_typelist.hpp"

/**
 * @internal
 *
 * Empty type list.
 */
struct EmptyTypeList {
  typedef bi::empty_easy_typelist type;
};

/**
 * @def EMPTY_TYPELIST
 *
 * Empty type list.
 *
 * @ingroup model_typelist
 */
#define EMPTY_TYPELIST EmptyTypeList::type::type

/**
 * @def BEGIN_TYPELIST(name)
 *
 * Open type list.
 *
 * @ingroup model_typelist
 *
 * @arg @c name Name to assign to the type list.
 */
#define BEGIN_TYPELIST(name) struct name { \
  typedef bi::empty_easy_typelist

/**
 * @def BEGIN_TYPELIST_TEMPLATE(name)
 *
 * Open templated type list. The macro should be immediately
 * preceded by a @c template statement.
 *
 * @ingroup model_typelist
 *
 * @arg @c name Name to assign to the type list definition.
 */
#define BEGIN_TYPELIST_TEMPLATE(name) struct name { \
  typedef typename bi::empty_easy_typelist

/**
 * @def SINGLE_TYPE(reps, single)
 *
 * Add single type to currently open type list.
 *
 * @ingroup model_typelist
 *
 * @arg @c reps Number of times to repeat this type.
 * @arg @c single Single type.
 */
#define SINGLE_TYPE(reps, single) ::push_back<single,reps>::type

/**
 * @def SINGLE_TYPE_TEMPLATE(reps, single, ...)
 *
 * Add templated single type to currently open type list.
 *
 * @ingroup model_typelist
 *
 * @arg @c reps Number of times to repeat this type.
 * @arg @c single Single type.
 * @arg Remaining variadic arguments are to accommodate template types
 * containing commas.
 */
#define SINGLE_TYPE_TEMPLATE(reps, single, ...) ::push_back<single,##__VA_ARGS__,reps>::type

/**
 * @def COMPOUND_TYPE(reps, compound)
 *
 * Add compound type to currently open type list.
 *
 * @ingroup model_typelist
 *
 * @arg @c reps Number of times to repeat this type.
 * @arg @c type Compound type.
 */
#define COMPOUND_TYPE(reps, compound) ::push_back_spec<compound,reps>::type

/**
 * @def COMPOUND_TYPE_TEMPLATE(reps, compound)
 *
 * Add compound type to currently open type list.
 *
 * @ingroup model_typelist
 *
 * @arg @c reps Number of times to repeat this type.
 * @arg @c type Compound type.
 * @arg Remaining variadic arguments are to accommodate template types
 * containing commas.
 */
#define COMPOUND_TYPE_TEMPLATE(reps, compound, ...) ::push_back_spec<compound,##__VA_ARGS__,reps>::type

/**
 * @def END_TYPELIST()
 *
 * Close type list.
 *
 * @ingroup model_typelist
 */
#define END_TYPELIST() type; \
};

/**
 * @def GET_TYPELIST(name)
 *
 * Retrieve previously defined type list.
 *
 * @ingroup model_typelist
 *
 * @arg @c name Name of the type list.
 */
#define GET_TYPELIST(name) name::type::type

/**
 * @def GET_TYPELIST_TEMPLATE(name)
 *
 * Retrieve previously defined templated type list.
 *
 * @ingroup model_typelist
 *
 * @arg @c name Name of the type list.
 * @arg Remaining variadic arguments are to accommodate template types
 * containing commas.
 */
#define GET_TYPELIST_TEMPLATE(name, ...) typename name,##__VA_ARGS__::type::type

#endif

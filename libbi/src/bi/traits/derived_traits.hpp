/**
 * @file
 *
 * Higher-level traits derived from other traits.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_DERIVED_TRAITS_HPP
#define BI_TRAITS_DERIVED_TRAITS_HPP

#include "likelihood_traits.hpp"
#include "prior_traits.hpp"
#include "random_traits.hpp"

namespace bi {
/**
 * @internal
 *
 * Log-variable.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_log_variable {
  static const bool value = is_log_normal_likelihood<X>::value || has_log_normal_prior<X>::value;
};

/**
 * @internal
 *
 * Implementation of all_gaussian_variates.
 */
template<class S>
struct all_gaussian_variates_impl {
  typedef typename front<S>::type head;
  typedef typename pop_front<S>::type tail;

  static const bool value = is_gaussian_variate<head>::value && all_gaussian_variates_impl<tail>::value;
};
template<>
struct all_gaussian_variates_impl<empty_typelist> {
  static const bool value = true;
};

/**
 * @internal
 *
 * Do all types in a type list have is_gaussian_variate?
 *
 * @ingroup model_trait
 *
 * @tparam S Type list.
 */
template<class S>
struct all_gaussian_variates {
  static const bool value = all_gaussian_variates_impl<S>::value;
};

/**
 * @internal
 *
 * Implementation of all_uniform_variates.
 */
template<class S>
struct all_uniform_variates_impl {
  typedef typename front<S>::type head;
  typedef typename pop_front<S>::type tail;

  static const bool value = is_uniform_variate<head>::value && all_uniform_variates_impl<tail>::value;
};
template<>
struct all_uniform_variates_impl<empty_typelist> {
  static const bool value = true;
};

/**
 * @internal
 *
 * Do all types in a type list have is_uniform_variate?
 *
 * @ingroup model_trait
 *
 * @tparam S Type list.
 */
template<class S>
struct all_uniform_variates {
  static const bool value = all_uniform_variates_impl<S>::value;
};

/**
 * @internal
 *
 * Implementation of all_wiener_increments.
 */
template<class S>
struct all_wiener_increments_impl {
  typedef typename front<S>::type head;
  typedef typename pop_front<S>::type tail;

  static const bool value = is_wiener_increment<head>::value && all_wiener_increments_impl<tail>::value;
};
template<>
struct all_wiener_increments_impl<empty_typelist> {
  static const bool value = true;
};

/**
 * @internal
 *
 * Do all types in a type list have is_wiener_increment?
 *
 * @ingroup model_trait
 *
 * @tparam S Type list.
 */
template<class S>
struct all_wiener_increments {
  static const bool value = all_wiener_increments_impl<S>::value;
};

/**
 * @internal
 *
 * Implementation of all_gaussian_likelihoods.
 */
template<class S>
struct all_gaussian_likelihoods_impl {
  typedef typename front<S>::type head;
  typedef typename pop_front<S>::type tail;

  static const bool value = is_gaussian_likelihood<head>::value && all_gaussian_likelihoods_impl<tail>::value;
};
template<>
struct all_gaussian_likelihoods_impl<empty_typelist> {
  static const bool value = true;
};

/**
 * @internal
 *
 * Do all types in a type list have is_gaussian_likelihood?
 *
 * @ingroup model_trait
 *
 * @tparam S Type list.
 */
template<class S>
struct all_gaussian_likelihoods {
  static const bool value = all_gaussian_likelihoods_impl<S>::value;
};

/**
 * @internal
 *
 * Implementation of all_log_normal_likelihoods.
 */
template<class S>
struct all_log_normal_likelihoods_impl {
  typedef typename front<S>::type head;
  typedef typename pop_front<S>::type tail;

  static const bool value = is_log_normal_likelihood<head>::value && all_log_normal_likelihoods_impl<tail>::value;
};
template<>
struct all_log_normal_likelihoods_impl<empty_typelist> {
  static const bool value = true;
};

/**
 * @internal
 *
 * Do all types in a type list have is_log_normal_likelihood?
 *
 * @ingroup model_trait
 *
 * @tparam S Type list.
 */
template<class S>
struct all_log_normal_likelihoods {
  static const bool value = all_log_normal_likelihoods_impl<S>::value;
};

}

#endif

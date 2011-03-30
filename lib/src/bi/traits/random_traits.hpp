/**
 * @file
 *
 * Traits related to random variates (r-nodes).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_TRAITS_RANDOM_TRAITS_HPP
#define BI_MODEL_TRAITS_RANDOM_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * Uniform random variate \f$u \sim \mathcal{U}[0,1)\f$.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_uniform_variate {
  static const bool value = false;
};

/**
 * @def IS_UNIFORM_VARIATE(X)
 *
 * Attach uniform trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_UNIFORM_VARIATE(X, ...) \
  namespace bi { \
  template<> \
  struct is_uniform_variate< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Gaussian random variate \f$u \sim \mathcal{N}(0,1)\f$.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_gaussian_variate {
  static const bool value = false;
};

/**
 * @def IS_GAUSSIAN_VARIATE(X)
 *
 * Attach Gaussian trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_GAUSSIAN_VARIATE(X, ...) \
  namespace bi { \
  template<> \
  struct is_gaussian_variate< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @def IS_NORMAL_VARIATE(X)
 *
 * Synonym for #IS_GAUSSIAN_VARIATE.
 *
 * @ingroup model_trait
 */
#define IS_NORMAL_VARIATE(X, ...) IS_GAUSSIAN_VARIATE(X, ##__VA_ARGS__)

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

/**
 * @internal
 *
 * Implementation of all_gaussian_variates, base case.
 */
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

/**
 * @internal
 *
 * Implementation of all_uniform_variates, base case.
 */
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

}

#endif

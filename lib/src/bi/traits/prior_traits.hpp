/**
 * @file
 *
 * Traits related to priors (s-, d-, c- and p-nodes).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_TRAITS_PRIOR_TRAITS_HPP
#define BI_MODEL_TRAITS_PRIOR_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * Gaussian prior.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_gaussian_prior {
  static const bool value = false;
};

/**
 * @def HAS_GAUSSIAN_PRIOR(X)
 *
 * Attach Gaussian prior trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_GAUSSIAN_PRIOR(X, ...) \
  namespace bi { \
  template<> \
  struct has_gaussian_prior< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @def HAS_NORMAL_PRIOR(X)
 *
 * @ingroup model_trait
 *
 * Synonym of #HAS_GAUSSIAN_PRIOR.
 */
#define HAS_NORMAL_PRIOR(X, ...) \
  HAS_GAUSSIAN_PRIOR(X, ##__VA_ARGS__)

/**
 * @internal
 *
 * Log-normal prior.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_log_normal_prior {
  static const bool value = false;
};

/**
 * @def HAS_LOG_NORMAL_PRIOR(X)
 *
 * Attach log-normal prior trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_LOG_NORMAL_PRIOR(X, ...) \
  namespace bi { \
  template<> \
  struct has_log_normal_prior< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Uniform prior.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_uniform_prior {
  static const bool value = false;
};

/**
 * @def HAS_UNIFORM_PRIOR(X)
 *
 * Attach uniform prior trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_UNIFORM_PRIOR(X, ...) \
  namespace bi { \
  template<> \
  struct has_uniform_prior< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

}

#endif

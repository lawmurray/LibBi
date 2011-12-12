/**
 * @file
 *
 * Traits related to likelihood calculations (o-nodes).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_TRAITS_LIKELIHOOD_TRAITS_HPP
#define BI_MODEL_TRAITS_LIKELIHOOD_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * Gaussian likelihood \f$u \sim \mathcal{N}(0,1)\f$.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_gaussian_likelihood {
  static const bool value = false;
};

/**
 * @def IS_GAUSSIAN_LIKELIHOOD(X)
 *
 * Attach Gaussian likelihood trait to node type.
 *
 * @ingroup model_trait
 *
 * The node type should then declare member function templates of the
 * types given in bi::GaussianLikelihoodMuFunction and
 * bi::GaussianLikelihoodSigmaFunction.
 *
 * @arg @c X Node type.
 */
#define IS_GAUSSIAN_LIKELIHOOD(X, ...) \
  namespace bi { \
  template<> \
  struct is_gaussian_likelihood< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @def IS_NORMAL_LIKELIHOOD(X)
 *
 * @ingroup model_trait
 *
 * Synonym for #IS_GAUSSIAN_LIKELIHOOD.
 */
#define IS_NORMAL_LIKELIHOOD(X, ...) IS_GAUSSIAN_LIKELIHOOD(X, ##__VA_ARGS__)

/**
 * @internal
 *
 * Log-normal likelihood.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_log_normal_likelihood {
  static const bool value = false;
};

/**
 * @def IS_LOG_NORMAL_LIKELIHOOD(X)
 *
 * Attach log-normal likelihood trait to node type.
 *
 * @ingroup model_trait
 *
 * The node type should then declare member function templates of the
 * types given in bi::LogNormalLikelihoodMuFunction and
 * bi::LogNormalLikelihoodSigmaFunction.
 *
 * @arg @c X Node type.
 */
#define IS_LOG_NORMAL_LIKELIHOOD(X, ...) \
  namespace bi { \
  template<> \
  struct is_log_normal_likelihood< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Trait for nodes with zero \f$\mu\f$ parameter.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_zero_mu {
  static const bool value = false;
};

/**
 * @def HAS_ZERO_MU(X)
 *
 * Attach zero \f$\mu\f$ trait to node.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_ZERO_MU(X, ...) \
  namespace bi { \
  template<> \
  struct has_zero_mu< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Trait for nodes with unit \f$\sigma\f$ parameter.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_unit_sigma {
  static const bool value = false;
};

/**
 * @def HAS_UNIT_SIGMA(X)
 *
 * Attach unit \f$\sigma\f$ trait to node.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_UNIT_SIGMA(X, ...) \
  namespace bi { \
  template<> \
  struct has_unit_sigma< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Trait for nodes with common \f$\sigma\f$ parameter across trajectories.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_common_sigma {
  static const bool value = false;
};

/**
 * @def HAS_COMMON_SIGMA(X)
 *
 * Attach common \f$\sigma\f$ trait to node.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_COMMON_SIGMA(X, ...) \
  namespace bi { \
  template<> \
  struct has_common_sigma< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

}

#endif

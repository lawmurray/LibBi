/**
 * @file
 *
 * Traits related to forward simulation (d-nodes and c-nodes).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_TRAITS_FORWARD_TRAITS_HPP
#define BI_MODEL_TRAITS_FORWARD_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * Generic forward function trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_generic_forward {
  static const bool value = false;
};

/**
 * @def IS_GENERIC_FORWARD(X)
 *
 * Attach generic forward function trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_GENERIC_FORWARD(X, ...) \
  namespace bi { \
  template<> \
  struct is_generic_forward< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * ODE forward function trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_ode_forward {
  static const bool value = false;
};

/**
 * @def IS_ODE_FORWARD(X)
 *
 * Attach ODE forward function trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_ODE_FORWARD(X, ...) \
  namespace bi { \
  template<> \
  struct is_ode_forward< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @def IS_SDE_FORWARD(X)
 *
 * Attach SDE forward function trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_SDE_FORWARD(X, ...) IS_ODE_FORWARD(X, ##__VA_ARGS__)

}

#endif

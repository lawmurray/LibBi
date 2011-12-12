/**
 * @file
 *
 * Traits related to static updates (s-nodes).
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_TRAITS_STATIC_TRAITS_HPP
#define BI_MODEL_TRAITS_STATIC_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * Generic static update trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_generic_static {
  static const bool value = false;
};

/**
 * @def IS_GENERIC_STATIC(X)
 *
 * Attach generic static trait to node type.
 *
 * @ingroup model_trait
 *
 * The node type should then declare a member function template of one of the
 * types given in GenericStaticFunction.
 *
 * @arg @c X Node type.
 */
#define IS_GENERIC_STATIC(X, ...) \
  namespace bi { \
  template<> \
  struct is_generic_static< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

}

#endif

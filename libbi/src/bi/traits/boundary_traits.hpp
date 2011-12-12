/**
 * @file
 *
 * Traits related to boundary conditions for dimensions.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_BOUNDARY_TRAITS_HPP
#define BI_TRAITS_BOUNDARY_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * Cyclic boundary condition on x-dimension.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_cyclic_x_boundary {
  static const bool value = false;
};

/**
 * @def HAS_CYCLIC_X_BOUNDARY(X)
 *
 * Attach cyclic boundary condition on x-dimension trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_CYCLIC_X_BOUNDARY(X, ...) \
  namespace bi { \
  template<> \
  struct has_cyclic_x_boundary< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Cyclic boundary condition on y-dimension.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_cyclic_y_boundary {
  static const bool value = false;
};

/**
 * @def HAS_CYCLIC_Y_BOUNDARY(X)
 *
 * Attach cyclic boundary condition on x-dimension trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_CYCLIC_Y_BOUNDARY(X, ...) \
  namespace bi { \
  template<> \
  struct has_cyclic_y_boundary< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Cyclic boundary condition on z-dimension.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_cyclic_z_boundary {
  static const bool value = false;
};

/**
 * @def HAS_CYCLIC_Z_BOUNDARY(X)
 *
 * Attach cyclic boundary condition on z-dimension trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_CYCLIC_Z_BOUNDARY(X, ...) \
  namespace bi { \
  template<> \
  struct has_cyclic_z_boundary< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Explicit boundary condition on x-dimension. This implies that boundary
 * conditions are encoded in the formulae.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_explicit_x_boundary {
  static const bool value = false;
};

/**
 * @def HAS_EXPLICIT_X_BOUNDARY(X)
 *
 * Attach explicit boundary condition on x-dimension trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_EXPLICIT_X_BOUNDARY(X, ...) \
  namespace bi { \
  template<> \
  struct has_explicit_x_boundary< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Explicit boundary condition on y-dimension. This implies that boundary
 * conditions are encoded in the formulae.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_explicit_y_boundary {
  static const bool value = false;
};

/**
 * @def HAS_EXPLICIT_Y_BOUNDARY(X)
 *
 * Attach explicit boundary condition on x-dimension trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_EXPLICIT_Y_BOUNDARY(X, ...) \
  namespace bi { \
  template<> \
  struct has_explicit_y_boundary< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * Explicit boundary condition on z-dimension. This implies that boundary
 * conditions are encoded in the formulae.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct has_explicit_z_boundary {
  static const bool value = false;
};

/**
 * @def HAS_EXPLICIT_Z_BOUNDARY(X)
 *
 * Attach explicit boundary condition on z-dimension trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define HAS_EXPLICIT_Z_BOUNDARY(X, ...) \
  namespace bi { \
  template<> \
  struct has_explicit_z_boundary< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

}

///@}

#endif

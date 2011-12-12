#ifndef DIMENSION_TRAITS_HPP
#define DIMENSION_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * Does node have x-dimension?
 *
 * @ingroup model_trait
 */
template<class X>
struct node_has_x {
  static const bool value = false;
};

/**
 * @internal
 *
 * Does node have y-dimension?
 *
 * @ingroup model_trait
 */
template<class X>
struct node_has_y {
  static const bool value = false;
};

/**
 * @internal
 *
 * Does node have z-dimension?
 *
 * @ingroup model_trait
 */
template<class X>
struct node_has_z {
  static const bool value = false;
};

}

/**
 * @internal
 *
 * @def HAS_X
 *
 * Set x-dimension for node.
 *
 * @ingroup model_trait
 */
#define HAS_X(X) \
  namespace bi { \
    template<> \
    struct node_has_x<X> { \
      static const bool value = true; \
    }; \
  }

/**
 * @internal
 *
 * @def HAS_Y
 *
 * Set y-dimension for node.
 *
 * @ingroup model_trait
 */
#define HAS_Y(X) \
  namespace bi { \
    template<> \
    struct node_has_y<X> { \
      static const bool value = true; \
    }; \
  }

/**
 * @internal
 *
 * @def HAS_Z
 *
 * Set z-dimension for node.
 *
 * @ingroup model_trait
 */
#define HAS_Z(X) \
  namespace bi { \
    template<> \
    struct node_has_z<X> { \
      static const bool value = true; \
    }; \
  }

#endif

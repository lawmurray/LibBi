/**
 * @file
 *
 * Node type traits.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_TRAITS_TYPE_TRAITS_HPP
#define BI_MODEL_TRAITS_TYPE_TRAITS_HPP

namespace bi {
/**
 * @internal
 *
 * S-node trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_s_node {
  static const bool value = false;
};

/**
 * @def IS_S_NODE(X)
 *
 * Attach s-node trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_S_NODE(X, ...) \
  namespace bi { \
  template<> \
  struct is_s_node< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * D-node trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_d_node {
  static const bool value = false;
};

/**
 * @def IS_D_NODE(X)
 *
 * Attach d-node trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_D_NODE(X, ...) \
  namespace bi { \
  template<> \
  struct is_d_node< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * C-node trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_c_node {
  static const bool value = false;
};

/**
 * @def IS_C_NODE(X)
 *
 * Attach c-node trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_C_NODE(X, ...) \
  namespace bi { \
  template<> \
  struct is_c_node< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * R-node trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_r_node {
  static const bool value = false;
};

/**
 * @def IS_R_NODE(X)
 *
 * Attach r-node trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_R_NODE(X, ...) \
  namespace bi { \
  template<> \
  struct is_r_node< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * F-node trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_f_node {
  static const bool value = false;
};

/**
 * @def IS_F_NODE(X)
 *
 * Attach f-node trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_F_NODE(X, ...) \
  namespace bi { \
  template<> \
  struct is_f_node< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * O-node trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_o_node {
  static const bool value = false;
};

/**
 * @def IS_O_NODE(X)
 *
 * Attach o-node trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_O_NODE(X, ...) \
  namespace bi { \
  template<> \
  struct is_o_node< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

/**
 * @internal
 *
 * P-node trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_p_node {
  static const bool value = false;
};

/**
 * @def IS_P_NODE(X)
 *
 * Attach p-node trait to node type.
 *
 * @ingroup model_trait
 *
 * @arg @c X Node type.
 */
#define IS_P_NODE(X, ...) \
  namespace bi { \
  template<> \
  struct is_p_node< X, ##__VA_ARGS__ > { \
    static const bool value = true; \
  }; }

}

#endif

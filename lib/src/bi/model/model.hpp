/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MODEL_MODEL_HPP
#define BI_MODEL_MODEL_HPP

#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../traits/type_traits.hpp"
#include "../traits/dimension_traits.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Node types.
 *
 * @ingroup model_low
 */
enum NodeType {
  /**
   * Static variable dependent on only parameters and other static
   * variables.
   */
  S_NODE,

  /**
   * Discrete-time variable.
   */
  D_NODE,

  /**
   * Continuous-time variable.
   */
  C_NODE,

  /**
   * Random variate.
   */
  R_NODE,

  /**
   * Forcing.
   */
  F_NODE,

  /**
   * Observable.
   */
  O_NODE,

  /**
   * Parameter.
   */
  P_NODE,

  /**
   * Number of node types.
   */
  NUM_NODE_TYPES
};

/**
 * Extended node types, continues NodeType enumeration.
 *
 * @ingroup model_low
 */
enum NodeExtType {
  /**
   * Random variate associated with o-node.
   */
  OR_NODE = NUM_NODE_TYPES,

  /**
   * Observation of o-node.
   */
  OY_NODE,

  /**
   * Number of extended node types.
   */
  NUM_EXT_NODE_TYPES
};

/**
 * Dimensions.
 *
 * @ingroup model_low
 */
enum Dimension {
  /**
   * X-dimension.
   */
  X_DIM,

  /**
   * Y-dimension.
   */
  Y_DIM,

  /**
   * Z-dimension.
   */
  Z_DIM,

  /**
   * Number of dimensions.
   */
  NUM_DIMENSIONS
};

/**
 * @internal
 *
 * Type list to which node belongs.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct node_typelist {
  /* select type list */
  typedef
    typename
    boost::mpl::if_<is_s_node<X>,
        typename B::STypeList,
    typename
    boost::mpl::if_<is_d_node<X>,
        typename B::DTypeList,
    typename
    boost::mpl::if_<is_c_node<X>,
        typename B::CTypeList,
    typename
    boost::mpl::if_<is_r_node<X>,
        typename B::RTypeList,
    typename
    boost::mpl::if_<is_f_node<X>,
        typename B::FTypeList,
    typename
    boost::mpl::if_<is_o_node<X>,
        typename B::OTypeList,
    typename
    boost::mpl::if_<is_p_node<X>,
        typename B::PTypeList,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type type;
};

/**
 * @internal
 *
 * Dimension length of node.
 *
 * @ingroup model_low
 *
 * @tparam HasDim Does node have this dimension?
 * @tparam DimLen Length of this dimension.
 */
template<int HasDim, int DimLen>
struct node_dimlen {
  static const int value = DimLen;
};

/**
 * @internal
 *
 * Base case of node_dimlen
 */
template<int DimLen>
struct node_dimlen<0,DimLen> {
  static const int value = 1;
};

/**
 * @internal
 *
 * Id (position) of node amongst own type in model.
 *
 * @ingroup model_low
 */
template<class B, class X>
struct node_id {
  static const int value = index<typename node_typelist<B,X>::type,X>::value;
};

/**
 * @internal
 *
 * Size of node along x-dimension.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct node_x_size {
  static const int value = node_dimlen<node_has_x<X>::value,B::NX>::value;
};

/**
 * @internal
 *
 * Size of node along y-dimension.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct node_y_size {
  static const int value = node_dimlen<node_has_y<X>::value,B::NY>::value;
};

/**
 * @internal
 *
 * Size of node along z-dimension.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct node_z_size {
  static const int value = node_dimlen<node_has_z<X>::value,B::NZ>::value;
};

/**
 * @internal
 *
 * Size of node type.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct node_size {
  static const int value = node_x_size<B,X>::value*
      node_y_size<B,X>::value*node_z_size<B,X>::value;
};

/**
 * @internal
 *
 * Recursion type for calculating starting index of node type.
 *
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam X Node type.
 */
template<class B, class S, class X>
struct node_start_impl {
  /**
   * Starting index of @p X in @p S.
   */
  static const int value = node_size<B,typename front<S>::type>::value +
      node_start_impl<B,typename pop_front<S>::type,X>::value;
};

/**
 * @internal
 *
 * Base case of node_start_impl
 */
template<class B, class S>
struct node_start_impl<B,S,typename front<S>::type> {
  static const int value = 0;
};

/**
 * @internal
 *
 * Starting index of node type in net.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct node_start {
  static const int value = node_start_impl<B,typename node_typelist<B,X>::type,X>::value;
};

/**
 * @internal
 *
 * Ending index of node type in net. Equivalent to @c node_start<B,X>::value +
 * node_size<X>::value.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct node_end {
  static const int value = node_start<B,X>::value +
      node_size<B,X>::value;
};

/**
 * @internal
 *
 * Recursion type for calculating size of net.
 *
 * @tparam B Model type.
 * @tparam S Type list.
 */
template<class B, class S>
struct net_size_impl {
  /**
   * Size of @p S.
   */
  static const int value = node_size<B,typename front<S>::type>::value +
      net_size_impl<B,typename pop_front<S>::type>::value;
};

/**
 * @internal
 *
 * Base case of net_size_impl
 */
template<class B>
struct net_size_impl<B,empty_typelist> {
  static const int value = 0;
};

/**
 * @internal
 *
 * Size of model (no. nodes).
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam S Type list.
 */
template<class B, class S>
struct net_size {
  static const int value = net_size_impl<B,S>::value;
};

/**
 * @internal
 *
 * @def Macro for parent declarations.
 */
#define MODEL_PA_DEC(nm) \
  /**
    @internal

    Get nm##-parent of node.

    @ingroup model_low

    @tparam X Type of child.
    @tparam I Index of nm##-parent.
   */ \
  template<class X, int I> \
  struct node_##nm##pa { \
    \
  };

MODEL_PA_DEC(s)
MODEL_PA_DEC(d)
MODEL_PA_DEC(c)
MODEL_PA_DEC(r)
MODEL_PA_DEC(f)
MODEL_PA_DEC(p)

/**
 * @internal
 *
 * @def Macro for parent dimension offset declarations.
 */
#define MODEL_OFFSET_DEC(nm, dim) \
  /**
    @internal

    Get dim##-offset of nm##-parent.

    @ingroup model_low
   */ \
  template<class X, int I> \
  struct node_##nm##off_##dim { \
    static const int value = 0; \
  };

MODEL_OFFSET_DEC(s, x)
MODEL_OFFSET_DEC(s, y)
MODEL_OFFSET_DEC(s, z)
MODEL_OFFSET_DEC(d, x)
MODEL_OFFSET_DEC(d, y)
MODEL_OFFSET_DEC(d, z)
MODEL_OFFSET_DEC(c, x)
MODEL_OFFSET_DEC(c, y)
MODEL_OFFSET_DEC(c, z)
MODEL_OFFSET_DEC(r, x)
MODEL_OFFSET_DEC(r, y)
MODEL_OFFSET_DEC(r, z)
MODEL_OFFSET_DEC(f, x)
MODEL_OFFSET_DEC(f, y)
MODEL_OFFSET_DEC(f, z)
MODEL_OFFSET_DEC(p, x)
MODEL_OFFSET_DEC(p, y)
MODEL_OFFSET_DEC(p, z)

/**
 * @internal
 *
 * @def SET_PA
 *
 * Set parent.
 *
 * @ingroup model_trait
 *
 * @arg nm Type of parent node (e.g. s, d, c, r, f, p).
 * @arg X Type of node.
 * @arg I Index of r-parent.
 * @arg Pa Type of parent.
 * @arg x Offset in x dimension.
 * @arg y Offset in y dimension.
 * @arg z Offset in z dimension.
 */
#define SET_PA(nm,X,I,Pa,x,y,z) \
  namespace bi { \
    template<> \
    struct node_##nm##pa<X,I> { \
      typedef Pa type; \
    }; \
    template<> \
    struct node_##nm##off_x<X,I> { \
      static const int value = x; \
    }; \
    template<> \
    struct node_##nm##off_y<X,I> { \
      static const int value = y; \
    }; \
    template<> \
    struct node_##nm##off_z<X,I> { \
      static const int value = z; \
    }; \
  }

}

///@}

#endif

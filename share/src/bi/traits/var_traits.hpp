/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TRAITS_VAR_TRAITS_HPP
#define BI_TRAITS_VAR_TRAITS_HPP

#include "../typelist/index.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../typelist/size.hpp"

#include "boost/mpl/if.hpp"

namespace bi {
/**
 * Variable types.
 *
 * @ingroup model_low
 */
enum VarType {
  /**
   * Noise (random) variable.
   */
  R_VAR,

  /**
   * State (dynamic) variable.
   */
  D_VAR,

  /**
   * Parameter.
   */
  P_VAR,

  /**
   * Input (forcing) variable.
   */
  F_VAR,

  /**
   * Observed variable.
   */
  O_VAR,

  /**
   * Auxiliary state (dynamic) variable.
   */
  DX_VAR,

  /**
   * Auxiliary parameter.
   */
  PX_VAR,

  /**
   * Built-in variable.
   */
  B_VAR,

  /**
   * Alternative noise variable.
   */
  RY_VAR,

  /**
   * Alternative state variable.
   */
  DY_VAR,

  /**
   * Alternative parameter.
   */
  PY_VAR,

  /**
   * Alternative observation.
   */
  OY_VAR
};


/**
 * Number of regular and built-in (not alternative) variable types.
 */
static const int NUM_VAR_TYPES = 8;

/**
 * Alternative type map.
 *
 * @ingroup model_low
 *
 * Maps a variable type to the type for its alternative buffer, or to itself
 * if no alternative exists.
 */
template<VarType T>
struct alt_type {
  static const VarType value = T;
};

/**
 * @internal
 */
template<>
struct alt_type<R_VAR> {
  static const VarType value = RY_VAR;
};

/**
 * @internal
 */
template<>
struct alt_type<D_VAR> {
  static const VarType value = DY_VAR;
};

/**
 * @internal
 */
template<>
struct alt_type<P_VAR> {
  static const VarType value = PY_VAR;
};

/**
 * @internal
 */
template<>
struct alt_type<O_VAR> {
  static const VarType value = OY_VAR;
};

/**
 * Id of variable.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct var_id {
  static const int value = X::ID;
};

/**
 * Size of variable type.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct var_size {
  static const int value = X::SIZE;
};

/**
 * Start of variable in the type list of its associated net.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct var_start {
  static const int value = X::START;
};

/**
 * Number of dimensions associated with variable type.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct var_num_dims {
  static const int value = X::NUM_DIMS;
};

/**
 * Get type of variable.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct var_type {
  static const VarType value = X::TYPE;
};

/**
 * State (dynamic) variable trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_d_var {
  static const bool value = var_type<X>::value == D_VAR;
};

/**
 * State (dynamic) auxiliary variable trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_dx_var {
  static const bool value = var_type<X>::value == DX_VAR;
};

/**
 * Noise (random) variable trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_r_var {
  static const bool value = var_type<X>::value == R_VAR;
};

/**
 * Input (forcing) variable trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_f_var {
  static const bool value = var_type<X>::value == F_VAR;
};

/**
 * Observation trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_o_var {
  static const bool value = var_type<X>::value == O_VAR;
};

/**
 * Parameter trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_p_var {
  static const bool value = var_type<X>::value == P_VAR;
};

/**
 * Auxiliary parameter trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_px_var {
  static const bool value = var_type<X>::value == PX_VAR;
};

/**
 * Built-in variable trait.
 *
 * @ingroup model_low
 *
 * @tparam X Variable type.
 */
template<class X>
struct is_b_var {
  static const bool value = var_type<X>::value == B_VAR;
};

/**
 * Common variable trait.
 */
template<class X>
struct is_common_var {
  static const bool value = is_f_var<X>::value || is_o_var<X>::value || is_p_var<X>::value || is_px_var<X>::value;
};

/**
 * Common alternative variable trait.
 */
template<class X>
struct is_common_var_alt {
  static const bool value = is_common_var<X>::value;
};

/**
 * Select parent type according to variable type. Used by #Pa.
 */
template<class V1, class V2, class V3, class V4, class X>
struct parent_type {
  /**
   * Parent type.
   */
  typedef
    typename
    boost::mpl::if_<is_p_var<X>,
        V1,
    typename
    boost::mpl::if_<is_px_var<X>,
        V1,
    typename
    boost::mpl::if_<is_f_var<X>,
        V2,
    typename
    boost::mpl::if_<is_r_var<X>,
        V3,
    typename
    boost::mpl::if_<is_d_var<X>,
        V4,
    typename
    boost::mpl::if_<is_dx_var<X>,
        V4,
    typename
    boost::mpl::if_<is_o_var<X>,
        V4,
    typename
    boost::mpl::if_<is_b_var<X>,
        V4,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type>::type type;
};

}

#endif

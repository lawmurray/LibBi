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
  OY_VAR,

  /**
   * Number of variable types.
   */
  NUM_VAR_TYPES
};

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
 * @tparam X Node type.
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
 * @tparam X Node type.
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
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct var_net_start {
  static const int value = X::START;
};

/**
 * End of variable in the type list of its associated net.
 *
 * @ingroup model_low
 *
 * @tparam B Model type.
 * @tparam X Node type.
 */
template<class B, class X>
struct var_net_end {
  static const int value = X::START + X::SIZE;
};

/**
 * Number of dimensions associated with variable type.
 *
 * @ingroup model_low
 *
 * @tparam X Node type.
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
 * @tparam X Node type.
 */
template<class X>
struct var_type {
  static const VarType value = X::TYPE;
};

/**
 * Should variable be output?
 *
 * @ingroup model_low
 *
 * @tparam X Node type.
 */
template<class X>
struct var_io {
  static const bool value = X::IO;
};

/**
 * State (dynamic) variable trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_d_var {
  static const bool value = var_type<X>::value == D_VAR;
};

/**
 * State (dynamic) auxiliary variable trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_dx_var {
  static const bool value = var_type<X>::value == DX_VAR;
};

/**
 * Noise (random) variable trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_r_var {
  static const bool value = var_type<X>::value == R_VAR;
};

/**
 * Input (forcing) variable trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_f_var {
  static const bool value = var_type<X>::value == F_VAR;
};

/**
 * Observation trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_o_var {
  static const bool value = var_type<X>::value == O_VAR;
};

/**
 * Parameter trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_p_var {
  static const bool value = var_type<X>::value == P_VAR;
};

/**
 * Auxiliary parameter trait.
 *
 * @ingroup model_trait
 *
 * @tparam X Node type.
 */
template<class X>
struct is_px_var {
  static const bool value = var_type<X>::value == PX_VAR;
};

}

#endif

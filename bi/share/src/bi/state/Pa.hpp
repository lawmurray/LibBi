/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_PA_HPP
#define BI_STATE_PA_HPP

namespace bi {
/**
 * Input (parents) access structure.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam T1 Return type.
 * @tparam V1 Access type for parameter and auxiliary parameters.
 * @tparam V2 Access type for input variables.
 * @tparam V3 Access type for noise variables.
 * @tparam V4 Access type for state, auxiliary state and observed
 * variables.
 */
template<Location L, class B, class T1, class V1, class V2, class V3,
    class V4>
struct Pa {
  /**
   * Scalar type.
   */
  typedef T1 value_type;

  /**
   * Get parent.
   *
   * @tparam X Node type.
   *
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  value_type fetch(const int p, const int ix) const;

  /**
   * Get parent from alternative buffer.
   *
   * @tparam X Node type.
   *
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  value_type fetch_alt(const int p, const int ix) const;
};

/**
 * Specialisation of Pa for host.
 */
template<class B, class T1, class V1, class V2, class V3, class V4>
struct Pa<ON_HOST,B,T1,V1,V2,V3,V4> {
  typedef T1 value_type;

  template<class X>
  CUDA_FUNC_HOST value_type fetch(const int p, const int ix) const;

  template<class X>
  CUDA_FUNC_HOST value_type fetch_alt(const int p, const int ix) const;
};

/**
 * Specialisation of Pa for device.
 */
template<class B, class T1, class V1, class V2, class V3, class V4>
struct Pa<ON_DEVICE,B,T1,V1,V2,V3,V4> {
  typedef T1 value_type;

  template<class X>
  CUDA_FUNC_DEVICE value_type fetch(const int p, const int ix) const;

  template<class X>
  CUDA_FUNC_DEVICE value_type fetch_alt(const int p, const int ix) const;
};

}

#include "boost/mpl/if.hpp"

template<class B, class T1, class V1, class V2, class V3, class V4>
template<class X>
inline T1 bi::Pa<bi::ON_HOST,B,T1,V1,V2,V3,V4>::fetch(const int p,
    const int ix) const {
  using namespace boost::mpl;

  /* select memory type */
  typedef
    typename
    if_<is_p_var<X>,
        V1,
    typename
    if_<is_px_var<X>,
        V1,
    typename
    if_<is_f_var<X>,
        V2,
    typename
    if_<is_r_var<X>,
        V3,
    typename
    if_<is_d_var<X>,
        V4,
    typename
    if_<is_dx_var<X>,
        V4,
    typename
    if_<is_o_var<X>,
        V4,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type V;

  return V::template fetch<B,X>(p, ix);
}

template<class B, class T1, class V1, class V2, class V3, class V4>
template<class X>
inline T1 bi::Pa<bi::ON_HOST,B,T1,V1,V2,V3,V4>::fetch_alt(const int p,
    const int ix) const {
  using namespace boost::mpl;

  /* select memory type */
  typedef
    typename
    if_<is_p_var<X>,
        V1,
    typename
    if_<is_px_var<X>,
        V1,
    typename
    if_<is_f_var<X>,
        V2,
    typename
    if_<is_r_var<X>,
        V3,
    typename
    if_<is_d_var<X>,
        V4,
    typename
    if_<is_dx_var<X>,
        V4,
    typename
    if_<is_o_var<X>,
        V4,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type V;

  return V::template fetch_alt<B,X>(p, ix);
}

template<class B, class T1, class V1, class V2, class V3, class V4>
template<class X>
inline T1 bi::Pa<bi::ON_DEVICE,B,T1,V1,V2,V3,V4>::fetch(
    const int p, const int ix) const {
  using namespace boost::mpl;

  /* select memory type */
  typedef
    typename
    if_<is_p_var<X>,
        V1,
    typename
    if_<is_px_var<X>,
        V1,
    typename
    if_<is_f_var<X>,
        V2,
    typename
    if_<is_r_var<X>,
        V3,
    typename
    if_<is_d_var<X>,
        V4,
    typename
    if_<is_dx_var<X>,
        V4,
    typename
    if_<is_o_var<X>,
        V4,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type V;

  return V::template fetch<B,X>(p, ix);
}

template<class B, class T1, class V1, class V2, class V3, class V4>
template<class X>
inline T1 bi::Pa<bi::ON_DEVICE,B,T1,V1,V2,V3,V4>::fetch_alt(
    const int p, const int ix) const {
  using namespace boost::mpl;

  /* select memory type */
  typedef
    typename
    if_<is_p_var<X>,
        V1,
    typename
    if_<is_px_var<X>,
        V1,
    typename
    if_<is_f_var<X>,
        V2,
    typename
    if_<is_r_var<X>,
        V3,
    typename
    if_<is_d_var<X>,
        V4,
    typename
    if_<is_dx_var<X>,
        V4,
    typename
    if_<is_o_var<X>,
        V4,
    /*else*/
        int
    /*end*/
    >::type>::type>::type>::type>::type>::type>::type V;

  return V::template fetch_alt<B,X>(p, ix);
}

#endif

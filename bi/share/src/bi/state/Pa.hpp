/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_PA_HPP
#define BI_STATE_PA_HPP

#include "State.hpp"

namespace bi {
/**
 * Input (parents) access structure.
 *
 * @ingroup state
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam V1 Access type for parameter and auxiliary parameters.
 * @tparam V2 Access type for input variables.
 * @tparam V3 Access type for noise variables.
 * @tparam V4 Access type for state, auxiliary state and observed
 * variables.
 */
template<Location L, class B, class V1, class V2, class V3, class V4>
struct Pa {
  //
};

/**
 * Specialisation of Pa for host.
 */
template<class B, class V1, class V2, class V3, class V4>
struct Pa<ON_HOST,B,V1,V2,V3,V4> {
  /**
   * Get parent variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  const typename parent_type<V1,V2,V3,V4,X>::type::vector_reference_type
  fetch(const State<B,ON_HOST>& s, const int p) const;

  /**
   * Get parent variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  const typename parent_type<V1,V2,V3,V4,X>::type::vector_reference_alt_type
  fetch_alt(const State<B,ON_HOST>& s, const int p) const;

  /**
   * Get parent variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  const typename parent_type<V1,V2,V3,V4,X>::type::value_type& fetch(
      const State<B,ON_HOST>& s, const int p, const int ix) const;

  /**
   * Get parent variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  const typename parent_type<V1,V2,V3,V4,X>::type::value_type& fetch_alt(
      const State<B,ON_HOST>& s, const int p, const int ix) const;
};

#ifdef ENABLE_CUDA
/**
 * Specialisation of Pa for device.
 */
template<class B, class V1, class V2, class V3, class V4>
struct Pa<ON_DEVICE,B,V1,V2,V3,V4> {
  /**
   * Get parent variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  CUDA_FUNC_DEVICE const typename parent_type<V1,V2,V3,V4,X>::type::vector_reference_type
  fetch(const State<B,ON_DEVICE>& s, const int p) const;

  /**
   * Get parent variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  CUDA_FUNC_DEVICE const typename parent_type<V1,V2,V3,V4,X>::type::vector_reference_alt_type
  fetch_alt(const State<B,ON_DEVICE>& s, const int p) const;

  /**
   * Get parent variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  CUDA_FUNC_DEVICE const typename parent_type<V1,V2,V3,V4,X>::type::value_type&
  fetch(const State<B,ON_DEVICE>& s, const int p, const int ix) const;

  /**
   * Get parent variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  CUDA_FUNC_DEVICE const typename parent_type<V1,V2,V3,V4,X>::type::value_type&
  fetch_alt(const State<B,ON_DEVICE>& s, const int p, const int ix) const;
};
#endif

}

template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::vector_reference_type bi::Pa<
  bi::ON_HOST,B,V1,V2,V3,V4>::fetch(const State<B,ON_HOST>& s,
  const int p) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch<B,X>(s, p);
}

template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::vector_reference_alt_type bi::Pa<
  bi::ON_HOST,B,V1,V2,V3,V4>::fetch_alt(const State<B,ON_HOST>& s,
  const int p) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch_alt<B,X>(s, p);
}

template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::value_type& bi::Pa<
  bi::ON_HOST,B,V1,V2,V3,V4>::fetch(const State<B,ON_HOST>& s, const int p,
  const int ix) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch<B,X>(s, p, ix);
}

template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::value_type& bi::Pa<
  bi::ON_HOST,B,V1,V2,V3,V4>::fetch_alt(const State<B,ON_HOST>& s,
  const int p, const int ix) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch_alt<B,X>(s, p, ix);
}

#ifdef ENABLE_CUDA
template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::vector_reference_type
bi::Pa<bi::ON_DEVICE,B,V1,V2,V3,V4>::fetch(
  const State<B,ON_DEVICE>& s, const int p) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch<B,X>(s, p);
}

template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::vector_reference_alt_type
bi::Pa<bi::ON_DEVICE,B,V1,V2,V3,V4>::fetch_alt(
  const State<B,ON_DEVICE>& s, const int p) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch_alt<B,X>(s, p);
}

template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::value_type& bi::Pa<bi::ON_DEVICE,B,V1,V2,V3,V4>::fetch(
  const State<B,ON_DEVICE>& s, const int p, const int ix) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch<B,X>(s, p, ix);
}

template<class B, class V1, class V2, class V3, class V4>
template<class X>
inline const typename bi::parent_type<V1,V2,V3,V4,X>::type::value_type& bi::Pa<bi::ON_DEVICE,B,V1,V2,V3,V4>::fetch_alt(
  const State<B,ON_DEVICE>& s, const int p, const int ix) const {
return parent_type<V1,V2,V3,V4,X>::type::template fetch_alt<B,X>(s, p, ix);
}
#endif

#endif

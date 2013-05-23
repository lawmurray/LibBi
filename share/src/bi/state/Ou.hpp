/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_OU_HPP
#define BI_STATE_OU_HPP

#include "State.hpp"

namespace bi {
/**
 * Output access structure.
 *
 * @ingroup state
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam V1 Access type of output.
 */
template<Location L, class B, class V1>
struct Ou {
  //
};

/**
 * Specialisation of Ou for host.
 */
template<class B, class V1>
struct Ou<ON_HOST,B,V1> {
  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  typename V1::vector_reference_type fetch(State<B,ON_HOST>& s, const int p);

  /**
   * Get variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  typename V1::vector_reference_type fetch_alt(State<B,ON_HOST>& s, const int p);

  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  typename V1::value_type& fetch(State<B,ON_HOST>& s, const int p, const int ix);

  /**
   * Get variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  typename V1::value_type& fetch_alt(State<B,ON_HOST>& s, const int p, const int ix);
};

#ifdef ENABLE_CUDA
/**
 * Specialisation of Ou for device.
 */
template<class B, class V1>
struct Ou<ON_DEVICE,B,V1> {
  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  CUDA_FUNC_DEVICE typename V1::vector_reference_type fetch(State<B,ON_DEVICE>& s, const int p);

  /**
   * Get variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   */
  template<class X>
  CUDA_FUNC_DEVICE typename V1::vector_reference_type fetch_alt(State<B,ON_DEVICE>& s, const int p);

  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  CUDA_FUNC_DEVICE typename V1::value_type& fetch(State<B,ON_DEVICE>& s, const int p, const int ix);

  /**
   * Get variable from alternative buffer.
   *
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   */
  template<class X>
  CUDA_FUNC_DEVICE typename V1::value_type& fetch_alt(State<B,ON_DEVICE>& s, const int p, const int ix);
};

#endif

}

template<class B, class V1>
template<class X>
inline typename V1::vector_reference_type bi::Ou<bi::ON_HOST,B,V1>::fetch(
    State<B,ON_HOST>& s, const int p) {
  return V1::template fetch<B,X>(s, p);
}

template<class B, class V1>
template<class X>
inline typename V1::vector_reference_type bi::Ou<bi::ON_HOST,B,V1>::fetch_alt(
    State<B,ON_HOST>& s, const int p) {
  return V1::template fetch_alt<B,X>(s, p);
}

template<class B, class V1>
template<class X>
inline typename V1::value_type& bi::Ou<bi::ON_HOST,B,V1>::fetch(State<B,ON_HOST>& s, const int p, const int ix) {
  return V1::template fetch<B,X>(s, p, ix);
}

template<class B, class V1>
template<class X>
inline typename V1::value_type& bi::Ou<bi::ON_HOST,B,V1>::fetch_alt(
    State<B,ON_HOST>& s, const int p, const int ix) {
  return V1::template fetch_alt<B,X>(s, p, ix);
}

#ifdef ENABLE_CUDA
template<class B, class V1>
template<class X>
inline typename V1::vector_reference_type bi::Ou<bi::ON_DEVICE,B,V1>::fetch(
    State<B,ON_DEVICE>& s, const int p) {
  return V1::template fetch<B,X>(s, p);
}

template<class B, class V1>
template<class X>
inline typename V1::vector_reference_type bi::Ou<bi::ON_DEVICE,B,V1>::fetch_alt(
    State<B,ON_DEVICE>& s, const int p) {
  return V1::template fetch_alt<B,X>(s, p);
}

template<class B, class V1>
template<class X>
inline typename V1::value_type& bi::Ou<bi::ON_DEVICE,B,V1>::fetch(
    State<B,ON_DEVICE>& s, const int p, const int ix) {
  return V1::template fetch<B,X>(s, p, ix);
}

template<class B, class V1>
template<class X>
inline typename V1::value_type& bi::Ou<bi::ON_DEVICE,B,V1>::fetch_alt(
    State<B,ON_DEVICE>& s, const int p, const int ix) {
  return V1::template fetch_alt<B,X>(s, p, ix);
}
#endif

#endif

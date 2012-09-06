/**
 * @file
 *
 * Functions for reading of state objects through main memory for SSE
 * instructions. Use host.hpp methods to bind.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_SSEHOST_HPP
#define BI_SSE_SSEHOST_HPP

#include "math/scalar.hpp"
#include "../host/host.hpp"

namespace bi {
/**
 * Facade for state as 128-bit SSE values in main memory.
 *
 * @ingroup state_host
 */
struct sse_host {
  typedef sse_real value_type;
  typedef host_vector_reference<sse_real> vector_reference_type;
  typedef typename host::vector_reference_alt_type vector_reference_alt_type;

  static const bool on_device = false;

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   *
   * @return Variable.
   */
  template<class B, class X>
  static vector_reference_type fetch(State<B,ON_HOST>& s, const int p);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   *
   * @return Value of the given variable.
   */
  template<class B, class X>
  static vector_reference_alt_type fetch_alt(State<B,ON_HOST>& s,
      const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   *
   * @return Variable.
   */
  template<class B, class X>
  static vector_reference_type fetch(const State<B,ON_HOST>& s, const int p);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   *
   * @return Value of the given variable.
   */
  template<class B, class X>
  static vector_reference_alt_type fetch_alt(const State<B,ON_HOST>& s,
      const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class B, class X>
  static sse_real& fetch(State<B,ON_HOST>& s, const int p, const int ix);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   *
   * @return Value of the given variable.
   */
  template<class B, class X>
  static bi::sse_real& fetch_alt(State<B,ON_HOST>& s, const int p,
      const int ix);

  /**
   * Fetch variable.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class B, class X>
  static const sse_real& fetch(const State<B,ON_HOST>& s, const int p,
      const int ix);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_host
   *
   * @tparam B Model type.
   * @tparam X Variable type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param ix Serial coordinate.
   *
   * @return Value of the given variable.
   */
  template<class B, class X>
  static const bi::sse_real& fetch_alt(const State<B,ON_HOST>& s, const int p,
      const int ix);
};

}

template<class B, class X>
inline bi::sse_host::vector_reference_type bi::sse_host::fetch(
    State<B,ON_HOST>& s, const int p) {
  /* pre-condition */
  BI_ASSERT(!is_common_var<X>::value);

  BOOST_AUTO(x, row(s.template getVar<X>(), p));
  return vector_reference_type(reinterpret_cast<sse_real*>(x.buf()), x.size(),
      x.inc());
}

template<class B, class X>
inline bi::sse_host::vector_reference_alt_type bi::sse_host::fetch_alt(
    State<B,ON_HOST>& s, const int p) {
  return host::template fetch_alt<B,X>(s, p);
}

template<class B, class X>
inline bi::sse_host::vector_reference_type bi::sse_host::fetch(
    const State<B,ON_HOST>& s, const int p) {
  /* pre-condition */
  BI_ASSERT(!is_common_var<X>::value);

  BOOST_AUTO(x, row(s.template getVar<X>(), p));
  return vector_reference_type(reinterpret_cast<sse_real*>(x.buf()), x.size(),
      x.inc());
}

template<class B, class X>
inline bi::sse_host::vector_reference_alt_type bi::sse_host::fetch_alt(
    const State<B,ON_HOST>& s, const int p) {
  return host::template fetch_alt<B,X>(s, p);
}

template<class B, class X>
inline bi::sse_real& bi::sse_host::fetch(State<B,ON_HOST>& s, const int p,
    const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var<X>::value);

  return *reinterpret_cast<sse_real*>(&s.template getVar<X>(p, ix));
}

template<class B, class X>
inline bi::sse_real& bi::sse_host::fetch_alt(State<B,ON_HOST>& s, const int p,
    const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var_alt<X>::value);

  return *reinterpret_cast<const sse_real*>(&s.template getVarAlt<X>(p, ix));
}

template<class B, class X>
inline const bi::sse_real& bi::sse_host::fetch(const State<B,ON_HOST>& s,
    const int p, const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var<X>::value);

  return *reinterpret_cast<const sse_real*>(&s.template getVar<X>(p, ix));
}

template<class B, class X>
inline const bi::sse_real& bi::sse_host::fetch_alt(const State<B,ON_HOST>& s,
    const int p, const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var_alt<X>::value);

  return *reinterpret_cast<const sse_real*>(&s.template getVarAlt<X>(p, ix));
}

#endif

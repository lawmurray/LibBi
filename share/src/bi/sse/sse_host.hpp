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
  typedef simd_real value_type;
  typedef host_vector_reference<simd_real> vector_reference_type;
  typedef host::vector_reference_alt_type vector_reference_alt_type;

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
  static simd_real& fetch(State<B,ON_HOST>& s, const int p, const int ix);

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
  static bi::simd_real& fetch_alt(State<B,ON_HOST>& s, const int p,
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
  static const simd_real& fetch(const State<B,ON_HOST>& s, const int p,
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
  static const bi::simd_real& fetch_alt(const State<B,ON_HOST>& s, const int p,
      const int ix);
};

/**
 * Load targets from state into contiguous vector.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam V1 Vector type.
 *
 * @param s State.
 * @param p Trajectory id.
 * @param[out] x Vector.
 */
template<class B, class S, class V1>
void sse_host_load(State<B,ON_HOST>& s, const int p, V1 x);

/**
 * Store targets from contiguous vector into state.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 *
 * @param s[out] State.
 * @param p Trajectory id.
 * @param x Vector.
 */
template<class B, class S, class V1>
void sse_host_store(State<B,ON_HOST>& s, const int p, const V1 x);

}

#include "sse_host_load_visitor.hpp"
#include "sse_host_store_visitor.hpp"

template<class B, class X>
inline bi::sse_host::vector_reference_type bi::sse_host::fetch(
    State<B,ON_HOST>& s, const int p) {
  /* pre-condition */
  BI_ASSERT(!is_common_var<X>::value);

  BOOST_AUTO(x, row(s.template getVar<X>(), p));
  return vector_reference_type(reinterpret_cast<simd_real*>(x.buf()), x.size(),
      x.inc()/BI_SIMD_SIZE);
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
  return vector_reference_type(reinterpret_cast<simd_real*>(x.buf()), x.size(),
      x.inc()/BI_SIMD_SIZE);
}

template<class B, class X>
inline bi::sse_host::vector_reference_alt_type bi::sse_host::fetch_alt(
    const State<B,ON_HOST>& s, const int p) {
  return host::template fetch_alt<B,X>(s, p);
}

template<class B, class X>
inline bi::simd_real& bi::sse_host::fetch(State<B,ON_HOST>& s, const int p,
    const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var<X>::value);

  return *reinterpret_cast<simd_real*>(&s.template getVar<X>(p, ix));
}

template<class B, class X>
inline bi::simd_real& bi::sse_host::fetch_alt(State<B,ON_HOST>& s, const int p,
    const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var_alt<X>::value);

  return *reinterpret_cast<simd_real*>(&s.template getVarAlt<X>(p, ix));
}

template<class B, class X>
inline const bi::simd_real& bi::sse_host::fetch(const State<B,ON_HOST>& s,
    const int p, const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var<X>::value);

  return *reinterpret_cast<const simd_real*>(&s.template getVar<X>(p, ix));
}

template<class B, class X>
inline const bi::simd_real& bi::sse_host::fetch_alt(const State<B,ON_HOST>& s,
    const int p, const int ix) {
  /* pre-condition */
  BI_ASSERT(!is_common_var_alt<X>::value);

  return *reinterpret_cast<const simd_real*>(&s.template getVarAlt<X>(p, ix));
}

template<class B, class S, class V1>
inline void bi::sse_host_load(State<B,ON_HOST>& s, const int p, V1 x) {
  sse_host_load_visitor<B,S,S>::accept(s, p, x);
}

template<class B, class S, class V1>
inline void bi::sse_host_store(State<B,ON_HOST>& s, const int p, const V1 x) {
  sse_host_store_visitor<B,S,S>::accept(s, p, x);
}

#endif

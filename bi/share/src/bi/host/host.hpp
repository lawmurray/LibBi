/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_HOST_HPP
#define BI_HOST_HOST_HPP

#include "../state/State.hpp"
#include "../math/vector.hpp"
#include "../misc/location.hpp"
#include "../traits/var_traits.hpp"

namespace bi {
/**
 * Facade for state in main memory.
 *
 * @ingroup state_host
 */
struct host {
  typedef real value_type;
  typedef host_vector_reference<real> vector_reference_type;
  typedef vector_reference_type vector_reference_alt_type;

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
  static vector_reference_type fetch_alt(State<B,ON_HOST>& s, const int p);

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
  static vector_reference_type fetch_alt(const State<B,ON_HOST>& s,
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
  static real& fetch(State<B,ON_HOST>& s, const int p, const int ix);

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
  static real& fetch_alt(State<B,ON_HOST>& s, const int p, const int ix);

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
  static const real& fetch(const State<B,ON_HOST>& s, const int p,
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
  static const real& fetch_alt(const State<B,ON_HOST>& s, const int p,
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
void host_load(State<B,ON_HOST>& s, const int p, V1 x);

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
void host_store(State<B,ON_HOST>& s, const int p, const V1 x);

}

#include "host_load_visitor.hpp"
#include "host_store_visitor.hpp"

template<class B, class X>
inline bi::host::vector_reference_type bi::host::fetch(State<B,ON_HOST>& s,
    const int p) {
  if (is_common_var<X>::value) {
    return row(s.template getVar<X>(), 0);
  } else {
    return row(s.template getVar<X>(), p);
  }
}

template<class B, class X>
inline bi::host::vector_reference_type bi::host::fetch_alt(
    State<B,ON_HOST>& s, const int p) {
  if (is_common_var_alt<X>::value) {
    return row(s.template getVarAlt<X>(), 0);
  } else {
    return row(s.template getVarAlt<X>(), p);
  }
}

template<class B, class X>
inline bi::host::vector_reference_type bi::host::fetch(
    const State<B,ON_HOST>& s, const int p) {
  if (is_common_var<X>::value) {
    return row(s.template getVar<X>(), 0);
  } else {
    return row(s.template getVar<X>(), p);
  }
}

template<class B, class X>
inline bi::host::vector_reference_type bi::host::fetch_alt(
    const State<B,ON_HOST>& s, const int p) {
  if (is_common_var_alt<X>::value) {
    return row(s.template getVarAlt<X>(), 0);
  } else {
    return row(s.template getVarAlt<X>(), p);
  }
}

template<class B, class X>
inline real& bi::host::fetch(State<B,ON_HOST>& s, const int p, const int ix) {
  return s.template getVar<X>(p, ix);
}

template<class B, class X>
inline real& bi::host::fetch_alt(State<B,ON_HOST>& s, const int p,
    const int ix) {
  return s.template getVarAlt<X>(p, ix);
}

template<class B, class X>
inline const real& bi::host::fetch(const State<B,ON_HOST>& s, const int p,
    const int ix) {
  return s.template getVar<X>(p, ix);
}

template<class B, class X>
inline const real& bi::host::fetch_alt(const State<B,ON_HOST>& s, const int p,
    const int ix) {
  return s.template getVarAlt<X>(p, ix);
}

template<class B, class S, class V1>
inline void bi::host_load(State<B,ON_HOST>& s, const int p, V1 x) {
  host_load_visitor<B,S,S>::accept(s, p, x);
}

template<class B, class S, class V1>
inline void bi::host_store(State<B,ON_HOST>& s, const int p, const V1 x) {
  host_store_visitor<B,S,S>::accept(s, p, x);
}

#endif

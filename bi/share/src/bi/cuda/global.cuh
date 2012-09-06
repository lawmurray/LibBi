/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_GLOBAL_CUH
#define BI_CUDA_GLOBAL_CUH

#include "cuda.hpp"
#include "../state/State.hpp"
#include "../math/vector.hpp"
#include "../misc/location.hpp"
#include "../traits/var_traits.hpp"

namespace bi {
/**
 * Facade for state in CUDA global memory.
 *
 * @ingroup state_gpu
 */
struct global {
  typedef real value_type;
  typedef gpu_vector_reference<real> vector_reference_type;
  typedef vector_reference_type vector_reference_alt_type;

  static const bool on_device = true;

  /**
   * Fetch variable.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE vector_reference_type fetch(State<B,ON_DEVICE>& s,
      const int p);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE vector_reference_type fetch_alt(
      State<B,ON_DEVICE>& s, const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE const vector_reference_type fetch(
      const State<B,ON_DEVICE>& s, const int p);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE const vector_reference_type fetch_alt(
      const State<B,ON_DEVICE>& s, const int p);

  /**
   * Fetch variable.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE real& fetch(State<B,ON_DEVICE>& s, const int p,
      const int ix);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE real& fetch_alt(State<B,ON_DEVICE>& s, const int p,
      const int ix);

  /**
   * Fetch variable.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE const real& fetch(const State<B,ON_DEVICE>& s,
      const int p, const int ix);

  /**
   * Fetch variable from alternative buffer.
   *
   * @ingroup state_gpu
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
  static CUDA_FUNC_DEVICE const real& fetch_alt(const State<B,ON_DEVICE>& s,
      const int p, const int ix);
};

}

template<class B, class X>
inline bi::global::vector_reference_type bi::global::fetch(
    State<B,ON_DEVICE>& s, const int p) {
  if (is_common_var<X>::value) {
    return row(s.template getVar<X>(), 0);
  } else {
    return row(s.template getVar<X>(), p);
  }
}

template<class B, class X>
inline bi::global::vector_reference_type bi::global::fetch_alt(
    State<B,ON_DEVICE>& s, const int p) {
  if (is_common_var_alt<X>::value) {
    return row(s.template getVarAlt<X>(), 0);
  } else {
    return row(s.template getVarAlt<X>(), p);
  }
}

template<class B, class X>
inline const bi::global::vector_reference_type bi::global::fetch(
    const State<B,ON_DEVICE>& s, const int p) {
  if (is_common_var<X>::value) {
    return row(s.template getVar<X>(), 0);
  } else {
    return row(s.template getVar<X>(), p);
  }
}

template<class B, class X>
inline const bi::global::vector_reference_type bi::global::fetch_alt(
    const State<B,ON_DEVICE>& s, const int p) {
  if (is_common_var_alt<X>::value) {
    return row(s.template getVarAlt<X>(), 0);
  } else {
    return row(s.template getVarAlt<X>(), p);
  }
}

template<class B, class X>
inline real& bi::global::fetch(State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  return s.template getVar<X>(p, ix);
}

template<class B, class X>
inline real& bi::global::fetch_alt(State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  return s.template getVarAlt<X>(p, ix);
}

template<class B, class X>
inline const real& bi::global::fetch(const State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  return s.template getVar<X>(p, ix);
}

template<class B, class X>
inline const real& bi::global::fetch_alt(const State<B,ON_DEVICE>& s,
    const int p, const int ix) {
  return s.template getVarAlt<X>(p, ix);
}

#endif

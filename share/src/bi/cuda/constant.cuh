/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_CONSTANT_CUH
#define BI_CUDA_CONSTANT_CUH

#include "cuda.hpp"
#include "../state/State.hpp"
#include "../math/vector.hpp"
#include "../misc/location.hpp"
#include "../traits/var_traits.hpp"

/**
 * Constant memory.
 */
static CUDA_VAR_CONSTANT real const_mem[1024];

namespace bi {
/**
 * Bind state to %constant memory.
 *
 * @ingroup state_gpu
 *
 * @tparam B Model type.
 *
 * @param s State.
 */
template<class B>
void const_bind(const State<B,ON_DEVICE>& s);

/**
 * Facade for node state in CUDA %constant memory.
 *
 * @ingroup state_gpu
 */
struct constant {
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
  static CUDA_FUNC_DEVICE vector_reference_type fetch(
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
   * @return Variable.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE vector_reference_type fetch_alt(
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
   * @return Variable.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE const real& fetch_alt(const State<B,ON_DEVICE>& s,
      const int p, const int ix);
};

}

template<class B>
void bi::const_bind(const State<B,ON_DEVICE>& s) {
  CUDA_CHECKED_CALL(cudaMemcpyToSymbolAsync(const_mem, s.getCommon().buf(),
      s.getCommon().size2()*sizeof(real), 0, cudaMemcpyDeviceToDevice, 0));
}

template<class B, class X>
bi::constant::vector_reference_type bi::constant::fetch(
    const State<B,ON_DEVICE>& s, const int p) {
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  if (is_p_var<X>::value) {
    return vector_reference_type(const_mem + start, size);
  } else if (is_px_var<X>::value) {
    return vector_reference_type(const_mem + B::NP + start, size);
  } else {
    //BI_ASSERT(is_f_var<X>::value);
    return vector_reference_type(const_mem + B::NP + B::NPX + start, size);
  }
}

template<class B, class X>
bi::constant::vector_reference_type bi::constant::fetch_alt(
    const State<B,ON_DEVICE>& s, const int p) {
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  if (is_p_var<X>::value) {
    return vector_reference_type(const_mem + B::NP + B::NPX + B::NF + start, size);
  } else {
    //BI_ASSERT(is_o_var<X>::value);
    return vector_reference_type(const_mem + B::NP + B::NPX + B::NF + B::NP + start, size);
  }
}

template<class B, class X>
const real& bi::constant::fetch(const State<B,ON_DEVICE>& s, const int p,
    const int ix) {
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  if (is_p_var<X>::value) {
    return const_mem[start + ix];
  } else if (is_px_var<X>::value) {
    return const_mem[B::NP + start + ix];
  } else {
    //BI_ASSERT(is_f_var<X>::value);
    return const_mem[B::NP + B::NPX + start + ix];
  }
}

template<class B, class X>
const real& bi::constant::fetch_alt(const State<B,ON_DEVICE>& s,
    const int p, const int ix) {
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  if (is_p_var<X>::value) {
    return const_mem[B::NP + B::NPX + B::NF + start + ix];
  } else {
    //BI_ASSERT(is_o_var<X>::value);
    return const_mem[B::NP + B::NPX + B::NF + B::NP + start + ix];
  }
}

#endif

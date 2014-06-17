/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_SPARSESTATICLOGDENSITY_HPP
#define BI_UPDATER_SPARSESTATICLOGDENSITY_HPP

namespace bi {
/**
 * Static log-density evaluator.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticLogDensity {
public:
  /**
   * Evaluate log-density.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param mask Sparsity mask.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask,
      V1 lp);

  /**
   * Evaluate log-density for single trajectory.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param mask Sparsity mask.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_HOST>& s, const int p,
      const Mask<ON_HOST>& mask, V1 lp);

  #ifdef __CUDACC__
  /**
   * Evaluate log-density.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param mask Sparsity mask.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask,
      V1 lp);

  /**
   * Evaluate log-density for single trajectory.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param mask Sparsity mask.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_DEVICE>& s, const int p,
      const Mask<ON_DEVICE>& mask, V1 lp);
  #endif
};
}

#include "../host/updater/SparseStaticLogDensityHost.hpp"
#ifdef ENABLE_SSE
#include "../sse/updater/SparseStaticLogDensitySSE.hpp"
#endif
#ifdef __CUDACC__
#include "../cuda/updater/SparseStaticLogDensityGPU.cuh"
#endif

template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensity<B,S>::logDensities(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask, V1 lp) {
  // in practice non-SSE version seems faster than SSE here
  //#ifdef ENABLE_SSE
  //if (s.size() % BI_SIMD_SIZE == 0) {
  //  SparseStaticLogDensitySSE<B,S>::logDensities(s, mask, lp);
  //} else {
  //  SparseStaticLogDensityHost<B,S>::logDensities(s, mask, lp);
  //}
  //#else
  SparseStaticLogDensityHost<B,S>::logDensities(s, mask, lp);
  //#endif
}

template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensity<B,S>::logDensities(State<B,ON_HOST>& s,
    const int p, const Mask<ON_HOST>& mask, V1 lp) {
  SparseStaticLogDensityHost<B,S>::logDensities(s, p, mask, lp);
}

#ifdef __CUDACC__
template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensity<B,S>::logDensities(State<B,ON_DEVICE>& s,
    const Mask<ON_DEVICE>& mask, V1 lp) {
  SparseStaticLogDensityGPU<B,S>::logDensities(s, mask, lp);
}

template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensity<B,S>::logDensities(State<B,ON_DEVICE>& s,
    const int p, const Mask<ON_DEVICE>& mask, V1 lp) {
  SparseStaticLogDensityGPU<B,S>::logDensities(s, p, mask, lp);
}
#endif

#endif

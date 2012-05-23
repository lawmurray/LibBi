/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1622 $
 * $Date: 2011-06-13 22:28:52 +0800 (Mon, 13 Jun 2011) $
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
   * Update state.
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
   * Update single trajectory.
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
#ifdef __CUDACC__
#include "../cuda/updater/SparseStaticLogDensityGPU.cuh"
#endif

template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensity<B,S>::logDensities(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask, V1 lp) {
  SparseStaticLogDensityHost<B,S>::logDensities(s, mask, lp);
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
    const Mask<ON_HOST>& mask, V1 lp) {
  SparseStaticLogDensityGPU<B,S>::logDensities(s, mask, lp);
}

template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensity<B,S>::logDensities(State<B,ON_DEVICE>& s,
    const int p, const Mask<ON_HOST>& mask, V1 lp) {
  SparseStaticLogDensityGPU<B,S>::logDensities(s, p, mask, lp);
}
#endif

#endif

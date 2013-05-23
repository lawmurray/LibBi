/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_SPARSESTATICMAXLOGDENSITY_HPP
#define BI_UPDATER_SPARSESTATICMAXLOGDENSITY_HPP

namespace bi {
/**
 * Static maximum log-density evaluator.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticMaxLogDensity {
public:
  /**
   * Evaluate maximum log-density.
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
  static void maxLogDensities(State<B,ON_HOST>& s, const Mask<ON_HOST>& mask,
      V1 lp);

  /**
   * Evaluate maximum log-density for single trajectory.
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
  static void maxLogDensities(State<B,ON_HOST>& s, const int p,
      const Mask<ON_HOST>& mask, V1 lp);

  #ifdef __CUDACC__
  /**
   * Evaluate maximum log-density.
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
  static void maxLogDensities(State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask,
      V1 lp);

  /**
   * Evaluate maximum log-density for single trajectory.
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
  static void maxLogDensities(State<B,ON_DEVICE>& s, const int p,
      const Mask<ON_DEVICE>& mask, V1 lp);
  #endif
};
}

#include "../host/updater/SparseStaticMaxLogDensityHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/SparseStaticMaxLogDensityGPU.cuh"
#endif

template<class B, class S>
template<class V1>
void bi::SparseStaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask, V1 lp) {
  SparseStaticMaxLogDensityHost<B,S>::maxLogDensities(s, mask, lp);
}

template<class B, class S>
template<class V1>
void bi::SparseStaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_HOST>& s,
    const int p, const Mask<ON_HOST>& mask, V1 lp) {
  SparseStaticMaxLogDensityHost<B,S>::maxLogDensities(s, p, mask, lp);
}

#ifdef __CUDACC__
template<class B, class S>
template<class V1>
void bi::SparseStaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_DEVICE>& s,
    const Mask<ON_DEVICE>& mask, V1 lp) {
  SparseStaticMaxLogDensityGPU<B,S>::maxLogDensities(s, mask, lp);
}

template<class B, class S>
template<class V1>
void bi::SparseStaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_DEVICE>& s,
    const int p, const Mask<ON_DEVICE>& mask, V1 lp) {
  SparseStaticMaxLogDensityGPU<B,S>::maxLogDensities(s, p, mask, lp);
}
#endif

#endif

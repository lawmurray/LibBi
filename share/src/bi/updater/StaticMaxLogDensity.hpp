/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_STATICMAXLOGDENSITY_HPP
#define BI_UPDATER_STATICMAXLOGDENSITY_HPP

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
class StaticMaxLogDensity {
public:
  /**
   * Evaluate maximum log-density.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   *
   * The maximum log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_HOST>& s, V1 lp);

  /**
   * Evaluate maximum log-density for single trajectory.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param[in,out] lp Log-density.
   *
   * The maximum log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_HOST>& s, const int p, V1 lp);

  #ifdef __CUDACC__
  /**
   * Evaluate maximum log-density.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   *
   * The maximum log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_DEVICE>& s, V1 lp);

  /**
   * Evaluate maximum log-density for single trajectory.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param[in,out] lp Log-density.
   *
   * The maximum log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void maxLogDensities(State<B,ON_DEVICE>& s, const int p, V1 lp);
  #endif
};
}

#include "../host/updater/StaticMaxLogDensityHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/StaticMaxLogDensityGPU.cuh"
#endif

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_HOST>& s, V1 lp) {
  StaticMaxLogDensityHost<B,S>::maxLogDensities(s, lp);
}

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_HOST>& s, const int p,
    V1 lp) {
  StaticMaxLogDensityHost<B,S>::maxLogDensities(s, p, lp);
}

#ifdef __CUDACC__
template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_DEVICE>& s, V1 lp) {
  StaticMaxLogDensityGPU<B,S>::maxLogDensities(s, lp);
}

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensity<B,S>::maxLogDensities(State<B,ON_DEVICE>& s,
    const int p, V1 lp) {
  StaticMaxLogDensityGPU<B,S>::maxLogDensities(s, p, lp);
}
#endif

#endif

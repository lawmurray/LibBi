/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_STATICLOGDENSITY_HPP
#define BI_UPDATER_STATICLOGDENSITY_HPP

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
class StaticLogDensity {
public:
  /**
   * Evaluate log-density.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_HOST>& s, V1 lp);

  /**
   * Evaluate log-density for single trajectory.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_HOST>& s, const int p, V1 lp);

  #ifdef __CUDACC__
  /**
   * Evaluate log-density.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_DEVICE>& s, V1 lp);

  /**
   * Evaluate log-density for single trajectory.
   *
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class V1>
  static void logDensities(State<B,ON_DEVICE>& s, const int p, V1 lp);
  #endif
};
}

#include "../host/updater/StaticLogDensityHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/StaticLogDensityGPU.cuh"
#endif

template<class B, class S>
template<class V1>
void bi::StaticLogDensity<B,S>::logDensities(State<B,ON_HOST>& s, V1 lp) {
  StaticLogDensityHost<B,S>::logDensities(s, lp);
}

template<class B, class S>
template<class V1>
void bi::StaticLogDensity<B,S>::logDensities(State<B,ON_HOST>& s, const int p,
    V1 lp) {
  StaticLogDensityHost<B,S>::logDensities(s, p, lp);
}

#ifdef __CUDACC__
template<class B, class S>
template<class V1>
void bi::StaticLogDensity<B,S>::logDensities(State<B,ON_DEVICE>& s, V1 lp) {
  StaticLogDensityGPU<B,S>::logDensities(s, lp);
}

template<class B, class S>
template<class V1>
void bi::StaticLogDensity<B,S>::logDensities(State<B,ON_DEVICE>& s,
    const int p, V1 lp) {
  StaticLogDensityGPU<B,S>::logDensities(s, p, lp);
}
#endif

#endif

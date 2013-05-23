/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_DYNAMICLOGDENSITY_HPP
#define BI_UPDATER_DYNAMICLOGDENSITY_HPP

namespace bi {
/**
 * Dynamic log-density evaluator.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicLogDensity {
public:
  /**
   * Evaluate log-density.
   *
   * @tparam T1 Scalar type.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class T1, class V1>
  static void logDensities(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      V1 lp);

  /**
   * Evaluate log-density for single trajectory.
   *
   * @tparam T1 Scalar type.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class T1, class V1>
  static void logDensities(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p, V1 lp);

  #ifdef __CUDACC__
  /**
   * Evaluate log-density.
   *
   * @tparam T1 Scalar type.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class T1, class V1>
  static void logDensities(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s,
      V1 lp);

  /**
   * Evaluate log-density for single trajectory.
   *
   * @tparam T1 Scalar type.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param[in,out] lp Log-density.
   *
   * The log density is <i>added to</i> @p lp.
   */
  template<class T1, class V1>
  static void logDensities(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s,
      const int p, V1 lp);
  #endif
};
}

#include "../host/updater/DynamicLogDensityHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/DynamicLogDensityGPU.cuh"
#endif

template<class B, class S>
template<class T1, class V1>
void bi::DynamicLogDensity<B,S>::logDensities(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s, V1 lp) {
  DynamicLogDensityHost<B,S>::logDensities(t1, t2, s, lp);
}

template<class B, class S>
template<class T1, class V1>
void bi::DynamicLogDensity<B,S>::logDensities(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s, const int p, V1 lp) {
  DynamicLogDensityHost<B,S>::logDensities(t1, t2, s, p, lp);
}

#ifdef __CUDACC__
template<class B, class S>
template<class T1, class V1>
void bi::DynamicLogDensity<B,S>::logDensities(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s, V1 lp) {
  DynamicLogDensityGPU<B,S>::logDensities(t1, t2, s, lp);
}

template<class B, class S>
template<class T1, class V1>
void bi::DynamicLogDensity<B,S>::logDensities(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s, const int p, V1 lp) {
  DynamicLogDensityGPU<B,S>::logDensities(t1, t2, s, p, lp);
}
#endif

#endif

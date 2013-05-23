/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_DYNAMICSAMPLER_HPP
#define BI_UPDATER_DYNAMICSAMPLER_HPP

namespace bi {
/**
 * Dynamic sampler.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicSampler {
public:
  /**
   * Sample state.
   *
   * @tparam T1 Scalar type.
   *
   * @param[in,out] rng Random number generator.
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   */
  template<class T1>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      State<B,ON_HOST>& s);

  /**
   * Sample single trajectory.
   *
   * @tparam T1 Scalar type.
   *
   * @param[in,out] rng Random number generator.
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  template<class T1>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      State<B,ON_HOST>& s, const int p);

  #ifdef __CUDACC__
  /**
   * Sample state.
   *
   * @tparam T1 Scalar type.
   *
   * @param[in,out] rng Random number generator.
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   */
  template<class T1>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      State<B,ON_DEVICE>& s);

  /**
   * Sample single trajectory.
   *
   * @tparam T1 Scalar type.
   *
   * @param[in,out] rng Random number generator.
   * @param t1 Start of interval.
   * @param t2 End of interval.
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  template<class T1>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      State<B,ON_DEVICE>& s, const int p);
  #endif
};
}

#include "../host/updater/DynamicSamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/DynamicSamplerGPU.cuh"
#endif

template<class B, class S>
template<class T1>
void bi::DynamicSampler<B,S>::samples(Random& rng, const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  DynamicSamplerHost<B,S>::samples(rng, t1, t2, s);
}

template<class B, class S>
template<class T1>
void bi::DynamicSampler<B,S>::samples(Random& rng, const T1 t1, const T1 t2,
    State<B,ON_HOST>& s,
    const int p) {
  DynamicSamplerHost<B,S>::samples(rng, t1, t2, s, p);
}

#ifdef __CUDACC__
template<class B, class S>
template<class T1>
void bi::DynamicSampler<B,S>::samples(Random& rng, const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  DynamicSamplerGPU<B,S>::samples(rng, t1, t2, s);
}

template<class B, class S>
template<class T1>
void bi::DynamicSampler<B,S>::samples(Random& rng, const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s, const int p) {
  DynamicSamplerGPU<B,S>::samples(rng, t1, t2, s, p);
}
#endif

#endif

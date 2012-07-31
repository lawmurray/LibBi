/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_STATICSAMPLER_HPP
#define BI_UPDATER_STATICSAMPLER_HPP

namespace bi {
/**
 * Static sampler.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticSampler {
public:
  /**
   * Sample state.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  static void samples(Random& rng, State<B,ON_HOST>& s);

  /**
   * Sample single trajectory.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  static void samples(Random& rng, State<B,ON_HOST>& s, const int p);

  #ifdef __CUDACC__
  /**
   * Sample state.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  static void samples(Random& rng, State<B,ON_DEVICE>& s);

  /**
   * Sample single trajectory.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param p Trajectory index.
   */
  static void samples(Random& rng, State<B,ON_DEVICE>& s, const int p);
  #endif
};
}

#include "../host/updater/StaticSamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/StaticSamplerGPU.cuh"
#endif

template<class B, class S>
void bi::StaticSampler<B,S>::samples(Random& rng, State<B,ON_HOST>& s) {
  StaticSamplerHost<B,S>::samples(rng, s);
}

template<class B, class S>
void bi::StaticSampler<B,S>::samples(Random& rng, State<B,ON_HOST>& s,
    const int p) {
  StaticSamplerHost<B,S>::samples(rng, s, p);
}

#ifdef __CUDACC__
template<class B, class S>
void bi::StaticSampler<B,S>::samples(Random& rng, State<B,ON_DEVICE>& s) {
  StaticSamplerGPU<B,S>::samples(rng, s);
}

template<class B, class S>
void bi::StaticSampler<B,S>::samples(Random& rng, State<B,ON_DEVICE>& s,
    const int p) {
  StaticSamplerGPU<B,S>::samples(rng, s, p);
}
#endif

#endif

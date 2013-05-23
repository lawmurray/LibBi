/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_SPARSESTATICSAMPLER_HPP
#define BI_UPDATER_SPARSESTATICSAMPLER_HPP

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
class SparseStaticSampler {
public:
  /**
   * Sample state.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param mask Sparsity mask.
   */
  static void samples(Random& rng, State<B,ON_HOST>& s,
      const Mask<ON_HOST>& mask);

  /**
   * Sample single trajectory.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param mask Sparsity mask.
   */
  static void samples(Random& rng, State<B,ON_HOST>& s, const int p,
      const Mask<ON_HOST>& mask);

  #ifdef __CUDACC__
  /**
   * Sample state.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param mask Sparsity mask.
   */
  static void samples(Random& rng, State<B,ON_DEVICE>& s,
      const Mask<ON_DEVICE>& mask);

  /**
   * Sample single trajectory.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param p Trajectory index.
   * @param mask Sparsity mask.
   */
  static void samples(Random& rng, State<B,ON_DEVICE>& s, const int p,
      const Mask<ON_DEVICE>& mask);
  #endif
};
}

#include "../host/updater/SparseStaticSamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/updater/SparseStaticSamplerGPU.cuh"
#endif

template<class B, class S>
void bi::SparseStaticSampler<B,S>::samples(Random& rng, State<B,ON_HOST>& s,
    const Mask<ON_HOST>& mask) {
  SparseStaticSamplerHost<B,S>::samples(rng, s);
}

template<class B, class S>
void bi::SparseStaticSampler<B,S>::samples(Random& rng, State<B,ON_HOST>& s,
    const int p, const Mask<ON_HOST>& mask) {
  SparseStaticSamplerHost<B,S>::samples(rng, s, p);
}

#ifdef __CUDACC__
template<class B, class S>
void bi::SparseStaticSampler<B,S>::samples(Random& rng, State<B,ON_DEVICE>& s,
    const Mask<ON_DEVICE>& mask) {
  SparseStaticSamplerGPU<B,S>::samples(rng, s);
}

template<class B, class S>
void bi::SparseStaticSampler<B,S>::samples(Random& rng, State<B,ON_DEVICE>& s,
    const int p, const Mask<ON_DEVICE>& mask) {
  SparseStaticSamplerGPU<B,S>::samples(rng, s, p);
}
#endif

#endif

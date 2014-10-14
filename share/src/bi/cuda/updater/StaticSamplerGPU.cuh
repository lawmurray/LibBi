/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICSAMPLERGPU_CUH
#define BI_CUDA_UPDATER_STATICSAMPLERGPU_CUH

#include "../../random/Random.hpp"
#include "../../state/State.hpp"

namespace bi {
/**
 * Static sampler, on device.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticSamplerGPU {
public:
  /**
   * @copydoc StaticSampler::samples(Random&, State<B,ON_DEVICE>&)
   */
  static void samples(Random& rng, State<B,ON_DEVICE>& s);
};
}

#include "StaticSamplerKernel.cuh"
#include "../device.hpp"

template<class B, class S>
void bi::StaticSamplerGPU<B,S>::samples(Random& rng, State<B,ON_DEVICE>& s) {
  const int N = (block_is_matrix<S>::value) ? block_count<S>::value : block_size<S>::value;
  const int P = s.size();
  dim3 Db, Dg;

  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;

  if (N > 0) {
    kernelStaticSampler<B,S><<<Dg,Db>>>(rng.devRngs, s);
    CUDA_CHECK;
  }
}

#endif

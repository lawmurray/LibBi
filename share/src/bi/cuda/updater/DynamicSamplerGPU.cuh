/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICSAMPLERGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICSAMPLERGPU_CUH

#include "../../random/Random.hpp"
#include "../../state/State.hpp"

namespace bi {
/**
 * Dynamic sampler, on device.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicSamplerGPU {
public:
  /**
   * @copydoc DynamicSampler::samples(Random&, const T1, const T1, State<B,ON_DEVICE>&)
   */
  template<class T1>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      State<B,ON_DEVICE>& s);
};
}

#include "DynamicSamplerKernel.cuh"
#include "../device.hpp"

template<class B, class S>
template<class T1>
void bi::DynamicSamplerGPU<B,S>::samples(Random& rng, const T1 t1,
    const T1 t2, State<B,ON_DEVICE>& s) {
  const int N = (block_is_matrix<S>::value) ? block_count<S>::value : block_size<S>::value;
  const int P = s.size();
  dim3 Db, Dg;

  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;

  if (N > 0) {
    kernelDynamicSampler<B,S><<<Dg,Db>>>(rng.devRngs, t1, t2, s);
    CUDA_CHECK;
  }
}

#endif

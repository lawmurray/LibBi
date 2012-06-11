/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2632 $
 * $Date: 2012-05-31 16:35:09 +0800 (Thu, 31 May 2012) $
 */
#ifndef BI_CUDA_UPDATER_STATICSAMPLERGPU_CUH
#define BI_CUDA_UPDATER_STATICSAMPLERGPU_CUH

#include "../../random/Random.hpp"
#include "../../state/State.hpp"
#include "../../method/misc.hpp"

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
#include "../bind.cuh"
#include "../device.hpp"

template<class B, class S>
void bi::StaticSamplerGPU<B,S>::samples(Random& rng, State<B,ON_DEVICE>& s) {
  static const int N = block_size<S>::value;
  const int P = s.size();
  dim3 Db, Dg;

  Db.x = std::min(deviceIdealThreadsPerBlock(), P);
  Dg.x = std::min(deviceIdealThreads()/Db.x, (P + Db.x - 1)/Db.x);

  if (N > 0) {
    bind(s);
    kernelStaticSampler<B,S><<<Dg,Db>>>(rng);
    CUDA_CHECK;
    unbind(s);
  }
}

#endif

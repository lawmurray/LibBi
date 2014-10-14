/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_STRATIFIEDRESAMPLERGPU_CUH
#define BI_CUDA_RESAMPLER_STRATIFIEDRESAMPLERGPU_CUH

#include "ResamplerGPU.cuh"

namespace bi {
/**
 * StratifiedResampler implementation on device.
 */
class StratifiedResamplerGPU: public ResamplerGPU {
public:
  /**
   * @copydoc StratifiedResampler::op
   */
  template<class V1, class V2>
  static void op(Random& rng, const V1 Ws, V2 Os, const int n);
};
}

#include "StratifiedResamplerKernel.cuh"
#include "../device.hpp"

template<class V1, class V2>
void bi::StratifiedResamplerGPU::op(Random& rng, const V1 Ws, V2 Os,
    const int n) {
  /* pre-condition */
  BI_ASSERT(Ws.size() == Os.size());

  typedef typename V1::value_type T1;
  typename sim_temp_vector<V1>::type alphas(n);

  const int P = Ws.size();

  rng.uniforms(alphas);

  dim3 Db, Dg;
  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;
  //deviceBalance1d(Db, Dg);

  kernelStratifiedResamplerOp<<<Dg,Db>>>(alphas, Ws, Os, n);
  CUDA_CHECK;
}

#endif

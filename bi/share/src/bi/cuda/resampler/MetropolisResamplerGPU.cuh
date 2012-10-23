/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_METROPOLISRESAMPLERGPU_CUH
#define BI_CUDA_RESAMPLER_METROPOLISRESAMPLERGPU_CUH

#include "MetropolisResamplerKernel.cuh"
#include "../device.hpp"

template<class V1, class V2>
void bi::MetropolisResamplerGPU::ancestors(Random& rng, const V1 lws, V2 as,
    int C) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  const int P = lws.size();
  dim3 Db, Dg;

  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;
  //deviceBalance1d(Db, Dg);

  kernelMetropolisResamplerAncestors<<<Dg,Db>>>(rng.devRngs, lws, as, C);
  CUDA_CHECK;
}

#endif

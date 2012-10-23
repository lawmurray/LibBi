/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_REJECTIONRESAMPLERGPU_CUH
#define BI_CUDA_RESAMPLER_REJECTIONRESAMPLERGPU_CUH

#include "RejectionResamplerKernel.cuh"
#include "../device.hpp"

template<class V1, class V2>
void bi::RejectionResamplerGPU::ancestors(Random& rng, const V1 lws, V2 as,
    const typename V1::value_type maxLogWeight) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  const int P = as.size();
  dim3 Db, Dg;

  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;
  //deviceBalance1d(Db, Dg);

  kernelRejectionResamplerAncestors<<<Dg,Db>>>(rng.devRngs, lws, as,
      maxLogWeight);
  CUDA_CHECK;
}

#endif

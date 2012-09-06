/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_METHOD_METROPOLISRESAMPLER_CUH
#define BI_CUDA_METHOD_METROPOLISRESAMPLER_CUH

#include "MetropolisResamplerKernel.cuh"

#include <limits>

template<class V1, class V2>
void bi::MetropolisResamplerDeviceImpl::ancestors(Random& rng, const V1 lws,
    V2 as, int C) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  const int P = lws.size();
  dim3 Db, Dg;
  size_t Ns;

  //Db.x = bi::min(128, (P + 29) / 30); // for S1070
  Db.x = bi::min(256, (P + 13) / 14); // for S2050
  Dg.x = (P + Db.x - 1) / Db.x;
  Ns = 0; //P*sizeof(typename V1::value_type);

  const int seed = rng.uniformInt(numeric_limits<int>::min(), numeric_limits<int>::max());

  kernelMetropolisResamplerAncestors<<<Dg,Db,Ns>>>(lws, seed, P, C, as);
  CUDA_CHECK;
}

#endif

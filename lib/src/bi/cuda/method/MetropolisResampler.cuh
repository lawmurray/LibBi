/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_METHOD_METROPOLISRESAMPLER_CUH
#define BI_CUDA_METHOD_METROPOLISRESAMPLER_CUH

#include "../kernel/MetropolisResamplerKernel.cuh"
#include "../math/temp_vector.hpp"

template<class V1, class V2>
void bi::MetropolisResamplerDeviceImpl::ancestors(const V1& lws, V2& as, Random& rng, int L) {
  const int P = lws.size();
  dim3 Db, Dg;
  size_t Ns;

  //Db.x = std::min(128, (P + 29) / 30); // for S1070
  Db.x = std::min(256, (P + 13) / 14); // for S2050
  Dg.x = (P + Db.x - 1) / Db.x;
  Ns = 0; //P*sizeof(typename V1::value_type);

  const int seed = rng.uniformInt(0, 1e9);

  //BOOST_AUTO(seeds, host_temp_vector<int>(P));
  //rng.uniformInts(seeds->buf(), seeds->size(), 0, std::pow(2, 30));
  //BOOST_AUTO(devSeeds, gpu_map_vector(*seeds));
  BOOST_AUTO(devAs, gpu_map_vector(as));

  kernelMetropolisResamplerAncestors<<<Dg,Db,Ns>>>(lws, seed, P, L, *devAs);
  CUDA_CHECK;

  synchronize();
  //delete seeds;
  //delete devSeeds;
  delete devAs;
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RANDOM_RANDOMGPU_CUH
#define BI_RANDOM_RANDOMGPU_CUH

#include "RandomKernel.cuh"
#include "../device.hpp"

template<class V1>
void bi::RandomGPU::uniforms(Random& rng, V1 x,
    const typename V1::value_type lower,
    const typename V1::value_type upper) {
  dim3 Db, Dg;
  Db.x = bi::min(x.size(), deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(x.size(), deviceIdealThreads()) + Db.x - 1) / Db.x;

  kernelUniforms<<<Dg,Db>>>(rng.devRngs, x, lower, upper);
  CUDA_CHECK;
}

template<class V1>
void bi::RandomGPU::gaussians(Random& rng, V1 x,
    const typename V1::value_type mu, const typename V1::value_type sigma) {
  dim3 Db, Dg;
  Db.x = bi::min(x.size(), deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(x.size(), deviceIdealThreads()) + Db.x - 1) / Db.x;

  kernelGaussians<<<Dg,Db>>>(rng.devRngs, x, mu, sigma);
  CUDA_CHECK;
}

template<class V1>
void bi::RandomGPU::gammas(Random& rng, V1 x,
    const typename V1::value_type alpha, const typename V1::value_type beta) {
  dim3 Db, Dg;
  Db.x = bi::min(x.size(), deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(x.size(), deviceIdealThreads()) + Db.x - 1) / Db.x;

  kernelGammas<<<Dg,Db>>>(rng.devRngs, x, alpha, beta);
  CUDA_CHECK;
}

template<class V1>
void bi::RandomGPU::poissons(Random& rng, V1 x,
    const typename V1::value_type lambda) {
  dim3 Db, Dg;
  Db.x = bi::min(x.size(), deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(x.size(), deviceIdealThreads()) + Db.x - 1) / Db.x;

  kernelPoissons<<<Dg,Db>>>(rng.devRngs, x, lambda);
  CUDA_CHECK;
}

template<class V1, class V2>
void bi::RandomGPU::binomials(Random& rng, V1 x,
    const typename V1::value_type n, const typename V2::value_type p) {
  dim3 Db, Dg;
  Db.x = bi::min(x.size(), deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(x.size(), deviceIdealThreads()) + Db.x - 1) / Db.x;

  kernelBinomials<<<Dg,Db>>>(rng.devRngs, n, p);
  CUDA_CHECK;
}

template<class V1, class V2>
void bi::RandomGPU::multinomials(Random& rng, const V1 lps, V2 xs) {
  BI_ERROR_MSG(false, "Not implemented on device");
}

#endif

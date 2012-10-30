/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_RESAMPLERGPU_CUH
#define BI_CUDA_RESAMPLER_RESAMPLERGPU_CUH

#include "ResamplerKernel.cuh"
#include "../../primitive/vector_primitive.hpp"
#include "../../math/temp_vector.hpp"
#include "../../misc/omp.hpp"

#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/fill.h"

template<class V1>
void bi::ResamplerGPU::permute(V1 as) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);

  const int P = as.size();
  temp_gpu_vector<int>::type is(P);
  thrust::fill(is.begin(), is.end(), -1);

  dim3 Dg, Db;
  Db.x = bi::min(128, P);
  Dg.x = (P + Db.x - 1)/Db.x;

  kernelResamplerPrePermute<<<Dg,Db>>>(as.buf(), is.buf(), P);
  CUDA_CHECK;

  kernelResamplerPermute<<<Dg,Db>>>(as.buf(), is.buf(), P);
  CUDA_CHECK;
}

template<class V1, class M1>
void bi::ResamplerGPU::copy(const V1 as, M1 X) {
  /* pre-condition */
  BI_ASSERT(as.size() <= X.size1());
  BI_ASSERT(V1::on_device);
  BI_ASSERT(M1::on_device);

//  const int P = X.size1();
  const int P = as.size();
  const int N = X.size2();

  dim3 Dg, Db;
  Db.x = bi::min(128, P);
  Dg.x = (P + Db.x - 1)/Db.x;
  Db.y = (128 + Db.x - 1)/Db.x;
  Dg.y = (N + Db.y - 1)/Dg.y;

  kernelResamplerCopy<<<Dg,Db>>>(as.buf(), X);
  CUDA_CHECK;
}

template<class V1, class V2>
void bi::ResamplerGPU::ancestorsToOffspring(const V1 as, V2 os) {
  /* pre-conditions */
  BI_ASSERT(os.size() == as.size());

  const int P = os.size();
  typename sim_temp_vector<V1>::type keys(2*P), values(2*P);

  /* keys will consist of ancestry in [0..P-1], and 0..P-1 in [P..2P-1],
   * ensuring that all particle indices are represented */
  thrust::copy(as.begin(), as.end(), keys.begin());
  thrust::sequence(keys.begin() + P, keys.end());

  /* values are 1 for indices originally from the ancestry, 0 for others */
  thrust::fill(values.begin(), values.begin() + P, 1);
  thrust::fill(values.begin() + P, values.end(), 0);

  /* sort all that by key */
  bi::sort_by_key(keys, values);

  /* reduce by key to get final offspring counts */
  thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(),
      keys.begin(), os.begin());

  /* post-condition */
  BI_ASSERT(thrust::reduce(os.begin(), os.end()) == P);
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_METHOD_RESAMPLER_CUH
#define BI_CUDA_METHOD_RESAMPLER_CUH

#include "../kernel/ResamplerKernel.cuh"
#include "../math/temp_vector.hpp"

#include "thrust/fill.h"

template<class V1>
void bi::ResamplerDeviceImpl::permute(V1& as) {
  /* pre-condition */
  assert (V1::on_device);

  const int P = as.size();

  static V1 is;
  is.resize(P);
  thrust::fill(is.begin(), is.end(), -1);

  dim3 Dg, Db;
  Db.x = std::min(128, P);
  Dg.x = (P + Db.x - 1) / Db.x;

  kernelResamplerPermute<<<Dg,Db>>>(as.buf(), is.buf(), P);
}

template<class V1, class M1>
void bi::ResamplerDeviceImpl::copy(const V1& as, M1 s) {
  /* pre-condition */
  assert (as.size() == static_cast<typename V1::size_type>(s.size1()));
  assert (M1::on_device);

  BOOST_AUTO(devAs, gpu_map_vector(as));

  const int P = s.size1();
  const int N = s.size2();

  dim3 Dg, Db;
  Db.x = std::min(128, P);
  Dg.x = (P + Db.x - 1) / Db.x;
  Db.y = (128 + Db.x - 1) / Db.x;
  Dg.y = (N + Db.y - 1) / Dg.y;

  kernelResamplerCopy<<<Dg,Db>>>(devAs->buf(), s);

  synchronize();
  delete devAs;
}

#endif

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
#include "../../math/sim_temp_vector.hpp"

template<class V1, class V2>
void bi::ResamplerGPU::ancestorsToOffspring(const V1 as, V2 os) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  const int P = as.size();
  os.clear();

  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), P);
  Dg.x = (P + Db.x - 1)/Db.x;

  kernelAncestorsToOffspring<<<Dg,Db>>>(as, os);
  CUDA_CHECK;
}

template<class V1, class V2>
void bi::ResamplerGPU::offspringToAncestors(const V1 os, V2 as) {
  /* pre-conditions */
  BI_ASSERT(os.size() == as.size());
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  typename sim_temp_vector<V1>::type Os(os.size());

  sum_inclusive_scan(os, Os);
  cumulativeOffspringToAncestors(Os, as);
}

template<class V1, class V2>
void bi::ResamplerGPU::offspringToAncestorsPermute(const V1 os, V2 as) {
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = as.size();
  int_vector_type is(P), cs(P);

  offspringToAncestorsPrePermute(os, cs, is);
  postPermute(cs, is, as);
}

template<class V1, class V2, class V3>
void bi::ResamplerGPU::offspringToAncestorsPrePermute(const V1 os, V2 as,
    V3 is) {
  /* pre-conditions */
  BI_ASSERT(sum_reduce(os) == as.size());
  BI_ASSERT(as.size() == is.size());
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);
  BI_ASSERT(V3::on_device);

  const int P = as.size();
  typename sim_temp_vector<V1>::type Os(os.size());

  sum_inclusive_scan(os, Os);
  cumulativeOffspringToAncestorsPrePermute(Os, as, is);
}

template<class V1, class V2>
void bi::ResamplerGPU::cumulativeOffspringToAncestors(const V1 Os, V2 as) {
  /* pre-condition */
  BI_ASSERT(*(Os.end() - 1) == as.size());
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  const int P = as.size();
  typename sim_temp_vector<V2>::type is(0); // just placeholder

  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), P);
  Dg.x = (P + Db.x - 1)/Db.x;

  kernelCumulativeOffspringToAncestors<<<Dg,Db>>>(Os, as, is,
      DISABLE_PRE_PERMUTE);
  CUDA_CHECK;
}

template<class V1, class V2>
void bi::ResamplerGPU::cumulativeOffspringToAncestorsPermute(const V1 Os,
    V2 as) {
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = as.size();
  int_vector_type is(P), cs(P);

  cumulativeOffspringToAncestorsPrePermute(Os, cs, is);
  postPermute(cs, is, as);
}

template<class V1, class V2, class V3>
void bi::ResamplerGPU::cumulativeOffspringToAncestorsPrePermute(const V1 Os,
    V2 as, V3 is) {
  /* pre-conditions */
  BI_ASSERT(*(Os.end() - 1) == as.size());
  BI_ASSERT(as.size() == is.size());
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);
  BI_ASSERT(V3::on_device);

  const int P = as.size();
  //set_elements(is, P); // in this particular case, kernel call does this

  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), P);
  Dg.x = (P + Db.x - 1)/Db.x;

  // this way is currently broken, due to scan sum on weights not giving
  // monotonic sequence; investigating, might be a hardware error
  //kernelCumulativeOffspringToAncestors<<<Dg,Db>>>(Os, as, is,
  //    ENABLE_PRE_PERMUTE);
  kernelCumulativeOffspringToAncestors<<<Dg,Db>>>(Os, as, is,
      DISABLE_PRE_PERMUTE);
  CUDA_CHECK;
  prePermute(as, is);
}

template<class V1>
void bi::ResamplerGPU::permute(V1 as) {
  typename sim_temp_vector<V1>::type is(as.size()), cs(as.size());
  cs = as;
  prePermute(cs, is);
  postPermute(cs, is, as);
}

template<class V1, class V2>
void bi::ResamplerGPU::prePermute(V1 as, V2 is) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);
  BI_ASSERT(as.size() == is.size());

  const int P = as.size();
  set_elements(is, P);

  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), P);
  Dg.x = (P + Db.x - 1)/Db.x;

  kernelResamplerPrePermute<<<Dg,Db>>>(as, is);
  CUDA_CHECK;
}

template<class V1, class V2, class V3>
void bi::ResamplerGPU::postPermute(const V1 as, const V2 is, V3 cs) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);
  BI_ASSERT(as.size() == is.size());
  BI_ASSERT(as.size() == cs.size());

  const int P = as.size();

  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), P);
  Dg.x = (P + Db.x - 1)/Db.x;

  kernelResamplerPostPermute<<<Dg,Db>>>(as, is, cs);
  CUDA_CHECK;
}

template<class V1, class M1>
void bi::ResamplerGPU::copy(const V1 as, M1 X) {
  /* pre-condition */
  BI_ASSERT(as.size() <= X.size1());
  BI_ASSERT(V1::on_device);
  BI_ASSERT(M1::on_device);

  const int P = as.size();
  const int N = X.size2();

  dim3 Dg, Db;
  Db.x = bi::min(deviceIdealThreadsPerBlock(), P);
  Dg.x = (P + Db.x - 1)/Db.x;
  Db.y = 1;
  Dg.y = N;

  kernelResamplerCopy<<<Dg,Db>>>(as, X);
  CUDA_CHECK;
}

#endif

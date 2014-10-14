/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_REJECTIONRESAMPLERGPU_CUH
#define BI_CUDA_RESAMPLER_REJECTIONRESAMPLERGPU_CUH

#include "ResamplerGPU.cuh"

namespace bi {
/**
 * RejectionResampler implementation on device.
 */
class RejectionResamplerGPU: public ResamplerGPU {
public:
  /**
   * @copydoc MultinomialResampler::ancestors
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);

  /**
   * @copydoc MultinomialResampler::ancestorsPermute
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);
};
}

#include "RejectionResamplerKernel.cuh"
#include "../device.hpp"

template<class V1, class V2>
void bi::RejectionResamplerGPU::ancestors(Random& rng, const V1 lws,
    V2 as, const typename V1::value_type maxLogWeight) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  typedef typename V2::value_type T2;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = as.size();
  const int bufSize = 4;

  int_vector_type is(0); // just placeholder

  dim3 Db, Dg;
  //int Ns;

  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;
  //deviceBalance1d(Db, Dg);

  //if (bufSize*Db.x*Dg.x <= P) {
  //  Ns = bufSize*Db.x*sizeof(T2);
  //  kernelRejectionResamplerAncestors2<<<Dg,Db,Ns>>>(rng.devRngs,
  //      lws, as, is, maxLogWeight, DISABLE_PRE_PERMUTE);
  //} else {
    kernelRejectionResamplerAncestors<<<Dg,Db>>>(rng.devRngs,
        lws, as, is, maxLogWeight, DISABLE_PRE_PERMUTE);
  //}
  CUDA_CHECK;
}

template<class V1, class V2>
void bi::RejectionResamplerGPU::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, const typename V1::value_type maxLogWeight) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  typedef typename V2::value_type T2;
  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = as.size();
  const int bufSize = 4;

  int_vector_type is(P), cs(P);
  set_elements(is, P);

  dim3 Db, Dg;
  //int Ns;

  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;
  //deviceBalance1d(Db, Dg);

  //if (bufSize*Db.x*Dg.x <= P) {
  //  Ns = bufSize*Db.x*sizeof(int);
  //  kernelRejectionResamplerAncestors2<<<Dg,Db,Ns>>>(rng.devRngs,
  //      lws, cs, is, maxLogWeight, ENABLE_PRE_PERMUTE);
  //} else {
      kernelRejectionResamplerAncestors<<<Dg,Db>>>(rng.devRngs,
          lws, cs, is, maxLogWeight, ENABLE_PRE_PERMUTE);
  //}
  CUDA_CHECK;

  /* finish permutation */
  postPermute(cs, is, as);
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_METROPOLISRESAMPLERGPU_CUH
#define BI_CUDA_RESAMPLER_METROPOLISRESAMPLERGPU_CUH

#include "ResamplerGPU.cuh"

namespace bi {
/**
 * MetropolisResampler implementation on device.
 */
class MetropolisResamplerGPU: public ResamplerGPU {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as, int B);

  /**
   * @copydoc MetropolisResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as, int B);
};
}

#include "MetropolisResamplerKernel.cuh"
#include "../device.hpp"

template<class V1, class V2>
void bi::MetropolisResamplerGPU::ancestors(Random& rng, const V1 lws, V2 as,
    int B) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();
  int_vector_type is(0); // just placeholder

  dim3 Db, Dg;

  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;
  //deviceBalance1d(Db, Dg);

  kernelMetropolisResamplerAncestors<<<Dg,Db>>>(rng.devRngs, lws, as, is, B,
      DISABLE_PRE_PERMUTE);
  CUDA_CHECK;
}

template<class V1, class V2>
void bi::MetropolisResamplerGPU::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, int B) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);
  BI_ASSERT(V2::on_device);

  typedef typename sim_temp_vector<V2>::type int_vector_type;

  const int P = lws.size();
  int_vector_type is(P), cs(P);
  set_elements(is, P);

  dim3 Db, Dg;
  Db.x = bi::min(P, deviceIdealThreadsPerBlock());
  Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;
  //deviceBalance1d(Db, Dg);

  kernelMetropolisResamplerAncestors<<<Dg,Db>>>(rng.devRngs, lws, cs, is, B,
      ENABLE_PRE_PERMUTE);
  CUDA_CHECK;

  /* finish permutation */
  postPermute(cs, is, as);
}

#endif

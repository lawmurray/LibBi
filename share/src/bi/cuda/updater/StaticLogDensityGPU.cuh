/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICLOGDENSITYGPU_CUH
#define BI_CUDA_UPDATER_STATICLOGDENSITYGPU_CUH

#include "../../state/State.hpp"

namespace bi {
/**
 * Static log-density updater, on device.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticLogDensityGPU {
public:
  template<class V1>
  static void logDensities(State<B,ON_DEVICE>& s, V1 lp);
};
}

#include "StaticLogDensityKernel.cuh"
#include "../device.hpp"

template<class B, class S>
template<class V1>
void bi::StaticLogDensityGPU<B,S>::logDensities(State<B,ON_DEVICE>& s,
    V1 lp) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);

  const int N = (block_is_matrix<S>::value) ? block_count<S>::value : block_size<S>::value;
  const int P = s.size();
  dim3 Db, Dg;

  Db.y = N;
  Dg.y = 1;
  Db.x = bi::max(1, bi::min(deviceIdealThreadsPerBlock()/N, P));
  Dg.x = (P + Db.x - 1)/Db.x;

  if (N > 0) {
    kernelStaticLogDensity<B,S><<<Dg,Db>>>(s, lp);
    CUDA_CHECK;
  }
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2639 $
 * $Date: 2012-06-01 04:52:16 +0000 (Fri, 01 Jun 2012) $
 */
#ifndef BI_CUDA_UPDATER_STATICMAXLOGDENSITYGPU_CUH
#define BI_CUDA_UPDATER_STATICMAXLOGDENSITYGPU_CUH

#include "../../state/State.hpp"
#include "../../method/misc.hpp"

namespace bi {
/**
 * Static maximum log-density updater, on device.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticMaxLogDensityGPU {
public:
  template<class V1>
  static void maxLogDensities(State<B,ON_DEVICE>& s, V1 lp);
};
}

#include "StaticMaxLogDensityKernel.cuh"
#include "../bind.cuh"
#include "../device.hpp"

template<class B, class S>
template<class V1>
void bi::StaticMaxLogDensityGPU<B,S>::maxLogDensities(State<B,ON_DEVICE>& s,
    V1 lp) {
  /* pre-condition */
  assert (V1::on_device);

  static const int N = block_size<S>::value;
  const int P = s.size();
  dim3 Db, Dg;

  Db.y = N;
  Dg.y = 1;
  Db.x = std::min(deviceIdealThreadsPerBlock()/N, P);
  Dg.x = (P + Db.x - 1)/Db.x;

  if (N > 0) {
    bind(s);
    kernelStaticMaxLogDensity<B,S><<<Dg,Db>>>(lp);
    CUDA_CHECK;
    unbind(s);
  }
}

#endif

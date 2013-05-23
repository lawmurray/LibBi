/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYGPU_HPP
#define BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYGPU_HPP

#include "../../state/State.hpp"

namespace bi {
/**
 * Sparse static log-density updater, on device.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticLogDensityGPU {
public:
  /**
   * @copydoc SparseStaticLogDensity::logDensities(State<B,ON_DEVICE>&, const Mask<ON_DEVICE>&, V1)
   */
  template<class V1>
  static void logDensities(State<B,ON_DEVICE>& s, const Mask<ON_DEVICE>& mask, V1 lp);
};
}

#include "SparseStaticLogDensityKernel.cuh"
#include "../device.hpp"

template<class B, class S>
template<class V1>
void bi::SparseStaticLogDensityGPU<B,S>::logDensities(State<B,ON_DEVICE>& s,
    const Mask<ON_DEVICE>& mask, V1 lp) {
  /* pre-condition */
  BI_ASSERT(V1::on_device);

  const int P = s.size();

  if (mask.size() > 0) {
    dim3 Dg, Db;

    Db.x = bi::min(deviceIdealThreadsPerBlock(), P); // over trajectories
    Dg.x = (P + Db.x - 1)/Db.x;

    kernelSparseStaticLogDensity<B,S><<<Dg,Db>>>(s, mask, lp);
    CUDA_CHECK;
  }
}

#endif

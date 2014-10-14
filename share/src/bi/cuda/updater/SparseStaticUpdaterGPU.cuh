/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICUPDATERGPU_CUH
#define BI_CUDA_UPDATER_SPARSESTATICUPDATERGPU_CUH

#include "../../state/State.hpp"
#include "../../state/Mask.hpp"

namespace bi {
/**
 * Sparse static updater, on device.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticUpdaterGPU {
public:
  static void update(State<B,ON_DEVICE>& s, Mask<ON_DEVICE>& mask);
};
}

#include "SparseStaticUpdaterKernel.cuh"
#include "../device.hpp"

template<class B, class S>
void bi::SparseStaticUpdaterGPU<B,S>::update(State<B,ON_DEVICE>& s,
    Mask<ON_DEVICE>& mask) {
  const int P = s.size();

  if (mask.size() > 0) {
    dim3 Dg, Db;

    Db.x = bi::min(deviceIdealThreadsPerBlock(), P); // over trajectories
    Dg.x = (P + Db.x - 1)/Db.x;

    kernelSparseStaticUpdater<B,S><<<Dg,Db>>>(s, mask);
    CUDA_CHECK;
  }
}

#endif

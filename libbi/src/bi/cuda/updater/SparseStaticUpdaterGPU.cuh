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
#include "../../buffer/Mask.hpp"
#include "../../method/misc.hpp"

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
#include "../bind.cuh"
#include "../device.hpp"

template<class B, class S>
void bi::SparseStaticUpdaterGPU<B,S>::update(State<B,ON_DEVICE>& s,
    Mask<ON_DEVICE>& mask) {
  const int P = s.size();

  if (mask.size() > 0) {
    dim3 Dg, Db;

    Db.x = std::min(deviceIdealThreadsPerBlock(), P); // over trajectories
    Dg.x = (P + Db.x - 1)/Db.x;

    bind(s);
    kernelSparseStaticUpdater<B,S><<<Dg,Db>>>(mask);
    CUDA_CHECK;
    unbind(s);
  }
}

#endif

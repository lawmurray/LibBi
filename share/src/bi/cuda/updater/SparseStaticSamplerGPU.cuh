/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICSAMPLERGPU_CUH
#define BI_CUDA_UPDATER_SPARSESTATICSAMPLERGPU_CUH

#include "../../random/Random.hpp"
#include "../../state/State.hpp"
#include "../../state/Mask.hpp"

namespace bi {
/**
 * Sparse static sampling, on device.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class SparseStaticSamplerGPU {
public:
  static void update(Random& rng, State<B,ON_DEVICE>& s,
      Mask<ON_DEVICE>& mask);
};
}

#include "SparseStaticSamplerKernel.cuh"
#include "../device.hpp"

template<class B, class S>
void bi::SparseStaticSamplerGPU<B,S>::update(Random& rng,
    State<B,ON_DEVICE>& s, Mask<ON_DEVICE>& mask) {
  const int P = s.size();

  if (mask.size() > 0) {
    dim3 Dg, Db;

    Db.x = bi::min(P, deviceIdealThreadsPerBlock());
    Dg.x = (bi::min(P, deviceIdealThreads()) + Db.x - 1)/Db.x;

    kernelSparseStaticSampler<B,S><<<Dg,Db>>>(rng.devRngs, s, mask);
    CUDA_CHECK;
  }
}

#endif

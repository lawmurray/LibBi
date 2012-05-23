/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICUPDATERGPU_CUH
#define BI_CUDA_UPDATER_STATICUPDATERGPU_CUH

#include "../../state/State.hpp"
#include "../../method/misc.hpp"

namespace bi {
/**
 * Static updater, on GPU.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class StaticUpdaterGPU {
public:
  static void update(State<B,ON_DEVICE>& s);
};
}

#include "SUpdateKernel.cuh"

template<class B, class S>
void bi::StaticUpdaterGPU<B,S>::update(State<B,ON_DEVICE>& s) {
  static const int N = block_size<S>::value;
  const int P = s.size();
  dim3 Db, Dg;
  size_t Ns;

  Db.x = std::min(128, P);
  Db.y = 1;
  Dg.x = (P + Db.x - 1)/Db.x;
  Dg.y = N;
  Ns = 0;

  if (N > 0) {
    kernelStaticUpdater<B,S><<<Dg,Db,Ns>>>();
    CUDA_CHECK;
  }
}

#endif

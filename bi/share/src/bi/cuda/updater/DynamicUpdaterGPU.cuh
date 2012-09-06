/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICUPDATERGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICUPDATERGPU_CUH

#include "../../state/State.hpp"
#include "../../method/misc.hpp"

namespace bi {
/**
 * Dynamic updater, on GPU.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DynamicUpdaterGPU {
public:
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s);
};
}

#include "DynamicUpdaterKernel.cuh"
#include "../bind.cuh"
#include "../device.hpp"

template<class B, class S>
template<class T1>
void bi::DynamicUpdaterGPU<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  static const int N = block_size<S>::value;
  const int P = s.size();
  dim3 Db, Dg;

  Db.x = bi::min(deviceIdealThreadsPerBlock(), P);
  Db.y = 1;
  Dg.x = (P + Db.x - 1)/Db.x;
  Dg.y = N;

  if (N > 0) {
    bind(s);
    kernelDynamicUpdater<B,S,T1><<<Dg,Db>>>(t1, t2, s);
    CUDA_CHECK;
    unbind(s);
  }
}

#endif

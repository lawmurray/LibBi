/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SUPDATER_CUH
#define BI_CUDA_UPDATER_SUPDATER_CUH
#include "../../method/misc.hpp"
#include "../kernel/SUpdateKernel.cuh"

template<class B, bi::StaticHandling SH>
void bi::SUpdater<B,SH>::update(Static<ON_DEVICE>& theta) {
  typedef typename B::STypeList S;

  dim3 Dg, Db;
  const int P = theta.size();
  if (SH == STATIC_SHARED) {
    Db.x = 1;
    Db.y = 1;
    Dg.x = 1;
    Dg.y = 1;
  } else {
    Db.x = std::min(256, P);
    Dg.x = (P + Db.x - 1) / Db.x;
    Db.y = 1;
    Dg.y = 1;
  }
  if (net_size<B,S>::value > 0) {
    bind(theta);
    kernelSUpdate<B,SH><<<Dg,Db>>>();
    CUDA_CHECK;
    unbind(theta);
  }
}

#endif

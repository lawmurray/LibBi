/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DUPDATER_CUH
#define BI_CUDA_UPDATER_DUPDATER_CUH

#include "../kernel/DUpdateKernel.cuh"
#include "../../math/misc.hpp"

#include <algorithm>

template<class B, bi::StaticHandling SH>
void bi::DUpdater<B,SH>::update(const real t, const real tnxt,
    State<bi::ON_DEVICE>& s) {
  typedef typename B::DTypeList S;
  static const int N = net_size<B,S>::value;

  /*
   * Execution config
   *
   * @li Using 256 or 512 in place of 128 seems to produce erroneous results
   * with both CUDA 2.3 and CUDA 3.0 beta about half the time.
   *
   * @li Eliminating shared memory use from DUpdater seems to have allowed
   * this to increase to 256, but still not 512. Still gives the occasional
   * error.
   */
  const int P = s.size();
  dim3 Db, Dg;
  size_t Ns;

  Db.x = std::min(std::max(512 / next_power_2(N), 1), std::min(512, P));
  Db.y = N;
  Dg.x = (P + Db.x - 1) / Db.x;
  Dg.y = 1;
  Ns = Db.x*Db.y*sizeof(real);

  if (N > 0) {
    bind(s);
    kernelDUpdate<B,SH><<<Dg,Db,Ns>>>(t, tnxt);
    CUDA_CHECK;
    unbind(s);
  }
}

#endif

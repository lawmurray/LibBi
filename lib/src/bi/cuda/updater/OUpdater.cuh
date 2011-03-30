/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_OUPDATER_CUH
#define BI_CUDA_UPDATER_OUPDATER_CUH

#include "../kernel/OUpdateKernel.cuh"

#include <algorithm>

template<class B, bi::StaticHandling SH>
template<class V1>
void bi::OUpdater<B,SH>::update(const V1& ids, State<ON_DEVICE>& s) {
  typedef typename B::OTypeList S;
  const int P = s.size();

  if (net_size<B,S>::value > 0) {
    /* execution config */
    dim3 Db, Dg;
    Db.x = std::min(256, P); // over trajectories
    Dg.x = (P + Db.x - 1) / Db.x;
    Db.y = 1; // over variables
    Dg.y = 1;

    /* launch */
    bind(s);
    if (V1::on_device) {
      kernelOUpdate<B,SH><<<Dg,Db>>>(ids.buf(), P);
      CUDA_CHECK;
    } else {
      BOOST_AUTO(devIds, gpu_map_vector(ids));
      *devIds = ids;
      kernelOUpdate<B,SH><<<Dg,Db>>>(devIds->buf(), P);
      CUDA_CHECK;
      synchronize();
      delete devIds;
    }
    unbind(s);
  }
}

#endif

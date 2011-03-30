/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_LUPDATER_CUH
#define BI_CUDA_UPDATER_LUPDATER_CUH

#include "../kernel/LUpdateKernel.cuh"
#include "../../math/misc.hpp"

template<class B>
template<class V1, class V2>
void bi::LUpdater<B>::update(State<ON_DEVICE>& s, const V1& ids, V2& lws) {
  const int P = lws.size();
  if (ids.size() > 0) {
    BOOST_AUTO(devIds, gpu_map_vector(ids));
    BOOST_AUTO(devLws, gpu_map_vector(lws));

    bind(s);
    dim3 Dg, Db;
    size_t Ns;

    Db.x = std::min(std::max(256 / nextPower2(ids.size()), 1),
        std::min(256, P)); // over trajectories
    Db.y = ids.size(); // over variables
    Dg.x = (P + Db.x - 1) / Db.x;
    Ns = Db.x*sizeof(real);

    kernelLUpdate<B><<<Dg,Db,Ns>>>(devIds->buf(), devLws->buf());
    CUDA_CHECK;
    lws = *devLws;

    unbind(s);
    synchronize();
    delete devIds;
    delete devLws;
  }
}

#endif

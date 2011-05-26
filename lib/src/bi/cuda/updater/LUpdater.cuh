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

template<class B, bi::StaticHandling SH>
template<class V1>
void bi::LUpdater<B,SH>::update(State<ON_DEVICE>& s, const SparseMask<ON_DEVICE>& mask,
    V1& lws) {
  typedef typename gpu_vector_map_type<V1>::type V2;
  typedef typename SparseMask<ON_DEVICE>::dense_block_type dense_block_type;
  typedef typename SparseMask<ON_DEVICE>::sparse_block_type sparse_block_type;

  const int P = lws.size();

  if (mask.size() > 0) {
    BOOST_AUTO(devLws, gpu_map_vector(lws));

    bind(s);
    dim3 Dg, Db;
    size_t Ns;
    int start = 0, size;

    /* update for each dense block, one by one */
    BOOST_AUTO(iter1, mask.getDenseMask().begin());
    while (iter1 != mask.getDenseMask().end()) {
      size = (*iter1)->size();
      if (size > 0) {
        Db.y = size; // over variables
        Dg.y = 1;
        Db.x = std::min(std::max(256 / next_power_2(Db.y), 1), std::min(256, P)); // over trajectories
        Dg.x = (P + Db.x - 1) / Db.x;
        Ns = Db.x*sizeof(real);

        kernelLUpdate<B,dense_block_type,V2,SH><<<Dg,Db,Ns>>>(**iter1, start, *devLws);
        CUDA_CHECK;
      }
      start += size;
      ++iter1;
    }

    /* update for each sparse block, one by one */
    BOOST_AUTO(iter2, mask.getSparseMask().begin());
    while (iter2 != mask.getSparseMask().end()) {
      size = (*iter2)->size();
      if (size > 0) {
        Db.y = size; // over variables
        Dg.y = 1;
        Db.x = std::min(std::max(256 / next_power_2(Db.y), 1), std::min(256, P)); // over trajectories
        Dg.x = (P + Db.x - 1) / Db.x;
        Ns = Db.x*sizeof(real);

        kernelLUpdate<B,sparse_block_type,V2,SH><<<Dg,Db,Ns>>>(**iter2, start, *devLws);
        CUDA_CHECK;
      }
      start += size;
      ++iter2;
    }

    lws = *devLws;
    unbind(s);
    if (!V1::on_device) {
      synchronize();
    }
    delete devLws;
  }
}

#endif

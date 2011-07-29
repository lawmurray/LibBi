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
void bi::OUpdater<B,SH>::update(const SparseMask<ON_DEVICE>& mask, State<ON_DEVICE>& s) {
  typedef typename SparseMask<ON_DEVICE>::dense_block_type dense_block_type;
  typedef typename SparseMask<ON_DEVICE>::sparse_block_type sparse_block_type;

  const int P = s.size();

  if (mask.size() > 0) {
    SparseMask<ON_DEVICE> devMask(mask);

    bind(s);
    dim3 Dg, Db;
    int start = 0;

    /* update for each dense block, one by one */
    BOOST_AUTO(iter1, devMask.getDenseMask().begin());
    while (iter1 != devMask.getDenseMask().end()) {
      Db.x = std::min(256, P); // over trajectories
      Dg.x = (P + Db.x - 1) / Db.x;
      Db.y = 1; // over variables
      Dg.y = 1;

      kernelOUpdate<B,dense_block_type,SH><<<Dg,Db>>>(**iter1, start, P);
      CUDA_CHECK;

      start += (*iter1)->size();
      ++iter1;
    }

    /* update for each sparse block, one by one */
    BOOST_AUTO(iter2, devMask.getSparseMask().begin());
    while (iter2 != devMask.getSparseMask().end()) {
      Db.x = std::min(256, P); // over trajectories
      Dg.x = (P + Db.x - 1) / Db.x;
      Db.y = 1; // over variables
      Dg.y = 1;

      kernelOUpdate<B,sparse_block_type,SH><<<Dg,Db>>>(**iter2, start, P);
      CUDA_CHECK;

      start += (*iter2)->size();
      ++iter2;
    }

    unbind(s);
  }
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICUPDATERKERNEL_CUH
#define BI_CUDA_UPDATER_SPARSESTATICUPDATERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for sparse static updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 *
 * @param mask Mask.
 */
template<class B, class S>
CUDA_FUNC_GLOBAL void kernelSparseStaticUpdater(const Mask<ON_DEVICE> mask);

}

#include "SparseStaticUpdaterVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S>
void bi::kernelSparseStaticUpdater(const Mask<ON_DEVICE> mask) {
  typedef Pa<ON_DEVICE,B,real,global,global,global,global> PX;
  typedef Ox<ON_DEVICE,B,real,global> OX;
  typedef SparseStaticUpdaterVisitorGPU<B,S,PX,OX> Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  /* update */
  if (p < constP) {
    Visitor::accept(mask, p, pax, x);
  }

}

#endif

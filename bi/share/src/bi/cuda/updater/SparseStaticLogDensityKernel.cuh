/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2636 $
 * $Date: 2012-05-31 20:44:30 +0800 (Thu, 31 May 2012) $
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYKERNEL_CUH
#define BI_CUDA_UPDATER_SPARSESTATICLOGDENSITYKERNEL_CUH

#include "../../buffer/Mask.hpp"
#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for sparse static log-density updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam V1 Vector type.
 *
 * @param mask Mask.
 * @param[in,out] lp Log-densities.
 */
template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void kernelSparseStaticLogDensity(
    const Mask<ON_DEVICE> mask, V1 lp);

}

#include "SparseStaticLogDensityVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"
#include "../../state/Ox.hpp"

template<class B, class S, class V1>
void bi::kernelSparseStaticLogDensity(const Mask<ON_DEVICE> mask, V1 lp) {
  typedef Pa<ON_DEVICE,B,real,global,global,global,global> PX;
  typedef Ox<ON_DEVICE,B,real,global> OX;
  typedef SparseStaticLogDensityVisitorGPU<B,S,PX,OX> Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  PX pax;
  OX x;

  /* update */
  if (p < constP) {
    Visitor::accept(mask, p, pax, x, lp(p));
  }
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2639 $
 * $Date: 2012-06-01 04:52:16 +0000 (Fri, 01 Jun 2012) $
 */
#ifndef BI_CUDA_UPDATER_STATICLOGDENSITYKERNEL_CUH
#define BI_CUDA_UPDATER_STATICLOGDENSITYKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for static log-density update.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam V1 Vector type.
 */
template<class B, class S, class V1>
CUDA_FUNC_GLOBAL void kernelStaticLogDensity(V1 lp);

}

#include "StaticLogDensityVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S, class V1>
void bi::kernelStaticLogDensity(V1 lp) {
  typedef Pa<ON_DEVICE,B,real,global,global,global,global> PX;
  typedef Ox<ON_DEVICE,B,real,global> OX;
  typedef StaticLogDensityVisitorGPU<B,S,PX,OX> Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = blockIdx.y*blockDim.y + threadIdx.y;
  PX pax;
  OX x;

  /* update */
  if (p < constP) {
    Visitor::accept(p, id, pax, x, lp(p));
  }
}

#endif

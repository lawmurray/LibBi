/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
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
  typedef typename V1::value_type T1;
  typedef Pa<ON_DEVICE,B,real,global,global,global,global> PX;
  typedef Ox<ON_DEVICE,B,real,global> OX;
  typedef StaticLogDensityVisitorGPU<B,S,PX,OX> Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = blockIdx.y*blockDim.y + threadIdx.y;
  PX pax;
  OX x;
  T1 lp1 = 0.0;

  /* compute local log-density */
  if (p < constP) {
    Visitor::accept(p, id, pax, x, lp1);
  }

  /* add to global log-density */
  for (int i = 0; i < block_size<S>::value; ++i) {
    __syncthreads();
    if (id == i && p < constP) {
      lp(p) += lp1;
    }
  }
}

#endif

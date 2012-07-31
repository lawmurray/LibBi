/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICUPDATERKERNEL_CUH
#define BI_CUDA_UPDATER_DYNAMICUPDATERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for dynamic update.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void kernelDynamicUpdater(const T1 t1, const T1 t2);

}

#include "DynamicUpdaterVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S, class T1>
void bi::kernelDynamicUpdater(const T1 t1, const T1 t2) {
  typedef Pa<ON_DEVICE,B,real,constant,global,global,global> PX;
  typedef Ox<ON_DEVICE,B,real,global> OX;
  typedef DynamicUpdaterVisitorGPU<B,S,T1,PX,OX> Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  PX pax;
  OX x;

  /* update */
  if (p < constP) {
    Visitor::accept(t1, t2, p, i, pax, x);
  }
}

#endif

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
CUDA_FUNC_GLOBAL void kernelDynamicUpdater(const T1 t1, const T1 t2,
    State<B,ON_DEVICE> s);

}

#include "DynamicUpdaterVisitorGPU.cuh"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void bi::kernelDynamicUpdater(const T1 t1, const T1 t2,
    State<B,ON_DEVICE> s) {
  typedef Pa<ON_DEVICE,B,constant,constant,global,global> PX;
  typedef Ou<ON_DEVICE,B,global> OX;
  typedef DynamicUpdaterVisitorGPU<B,S,T1,PX,OX> Visitor;

  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  PX pax;
  OX x;

  /* update */
  if (p < s.size()) {
    Visitor::accept(t1, t2, s, p, i, pax, x);
  }
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_RK4KERNELGPU_CUH
#define BI_CUDA_ODE_RK4KERNELGPU_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * Kernel function for RK4IntegratorGPU.
 *
 * @tparam B Model type.
 *
 * @param t1 Current time.
 * @param t2 Time to which to integrate.
 */
template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void kernelRK4(const T1 t1, const T1 t2);

}

#include "RK4VisitorGPU.cuh"
#include "IntegratorConstants.cuh"
#include "../constant.cuh"
#include "../shared.cuh"
#include "../global.cuh"

template<class B, class S, class T1>
void bi::kernelRK4(const T1 t1, const T1 t2) {
  typedef Pa<ON_DEVICE,B,real,constant,global,global,shared<S> > PX;
  typedef RK4VisitorGPU<B,S,S,real,PX,real> Visitor;

  /* indices */
  const int i = threadIdx.y; // variable index
  const int p = blockIdx.x*blockDim.x + threadIdx.x; // trajectory index
  const int q = i*blockDim.x + threadIdx.x; // shared memory index

  /* shared memory */
  real* __restrict__ xs = shared_mem;
  real& x = xs[q];
  PX pax;

  /* initialise */
  shared_init<B,S>(p, i);
  real t = t1, h = h0, x0 = x, x1, x2, x3, x4;
  __syncthreads();

  while (t < t2) {
    if (t + BI_REAL(1.01)*h - t2 > BI_REAL(0.0)) {
      h = t2 - t;
      if (h <= BI_REAL(0.0)) {
        t = t2;
        break;
      }
    }

    /* stages */
    Visitor::stage1(t, h, p, i, pax, x0, x1, x2, x3, x4);
    __syncthreads();
    x = x1;
    __syncthreads();

    Visitor::stage2(t, h, p, i, pax, x0, x2, x3, x4);
    __syncthreads();
    x = x2;
    __syncthreads();

    Visitor::stage3(t, h, p, i, pax, x0, x3, x4);
    __syncthreads();
    x = x3;
    __syncthreads();

    Visitor::stage4(t, h, p, i, pax, x0, x4);
    __syncthreads();
    x = x4;
    __syncthreads();

    x0 = x4;
    t += h;
  }

  /* commit back to global memory */
  shared_commit<B,S>(p, i);
}

#endif

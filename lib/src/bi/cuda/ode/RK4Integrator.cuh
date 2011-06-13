/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1292 $
 * $Date: 2011-02-22 13:45:22 +0800 (Tue, 22 Feb 2011) $
 */
#ifndef BI_CUDA_ODE_RK4INTEGRATOR_CUH
#define BI_CUDA_ODE_RK4INTEGRATOR_CUH

#include "RK4Visitor.cuh"
#include "IntegratorConstants.cuh"
#include "../../state/Pa.hpp"
#include "../constant.cuh"
#include "../shared.cuh"
#include "../texture.cuh"
#include "../global.cuh"
#include "../../traits/forward_traits.hpp"
#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, unsigned SH>
void bi::RK4Integrator<bi::ON_DEVICE,B,SH>::stepTo(const real tcur, const real tnxt) {
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
  typedef Pa<ON_DEVICE,B,real,pa,global,texture,pa,texture,shared,global> V1;
  #else
  typedef Pa<ON_DEVICE,B,real,pa,global,global,pa,global,shared,global> V1;
  #endif
  typedef RK4Visitor<ON_DEVICE,B,S,real,V1,real> Visitor;

  /* indices */
  const int id = threadIdx.y;
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int q = id*blockDim.x + threadIdx.x; // thread id

  /* shared memory */
  real* __restrict__ xs = shared_mem;

  /* refs for this thread */
  real& x = xs[q];
  real t, h;
  V1 pax(p);

  /* initialise */
  t = tcur;
  h = h0;

  /* integrate */
  real x0, x1, x2, x3, x4;

  #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
  x0 = tex2D(texCState, pax.p, id);
  #else
  x0 = globalCState(pax.p, id);
  #endif
  x = x0;
  __syncthreads();

  while (t < tnxt) {
    /* initialise */
    if (t + REAL(1.01)*h - tnxt > REAL(0.0)) {
      h = tnxt - t;
      if (h <= REAL(0.0)) {
        t = tnxt;
        break;
      }
    }

    /* stages */
    Visitor::stage1(t, h, pax, x0, x1, x2, x3, x4);
    __syncthreads();
    x = x1;
    __syncthreads();

    Visitor::stage2(t, h, pax, x0, x2, x3, x4);
    __syncthreads();
    x = x2;
    __syncthreads();

    Visitor::stage3(t, h, pax, x0, x3, x4);
    __syncthreads();
    x = x3;
    __syncthreads();

    Visitor::stage4(t, h, pax, x0, x4);
    __syncthreads();
    x = x4;
    __syncthreads();

    x0 = x4;
    t += h;
  }

  /* write result for this trajectory */
  globalCState(pax.p, id) = x;
}

#endif

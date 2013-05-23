/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_DOPRI5KERNELGPU_CUH
#define BI_CUDA_ODE_DOPRI5KERNELGPU_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for DOPRI5IntegratorGPU.
 *
 * @tparam B Model type.
 *
 * @param t1 Current time.
 * @param t2 Time to which to integrate.
 * @param[in,out] s State.
 */
template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void kernelDOPRI5(const T1 t1, const T1 t2,
    State<B,ON_DEVICE> s);

}

#include "DOPRI5VisitorGPU.cuh"
#include "IntegratorConstants.cuh"
#include "../constant.cuh"
#include "../shared.cuh"
#include "../global.cuh"

template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void bi::kernelDOPRI5(const T1 t1, const T1 t2, State<B,ON_DEVICE> s) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef DOPRI5VisitorGPU<B,S,S,real,PX,real> Visitor;

  /* sizes */
  const int P = s.size();
  const int N = block_size<S>::value;

  /* indices */
  const int i = threadIdx.y; // variable index
  const int p = blockIdx.x*blockDim.x + threadIdx.x; // trajectory index
  const int q = threadIdx.x; // trajectory index in block

  /* shared memory */
  real* __restrict__ ts = shared_mem;
  real* __restrict__ hs = ts + blockDim.x;
  real* __restrict__ e2s = hs + blockDim.x;
  real* __restrict__ logfacolds = e2s + blockDim.x;
  CUDA_VAR_SHARED bool done;

  /* refs for this thread */
  real& x = global_load_visitor<B,S,S>::accept(s, p, i);
  real& t = ts[q];
  real& h = hs[q];
  real& e2 = e2s[q];
  real& logfacold = logfacolds[q];
  PX pax;

  /* flags */
  bool k1in = false;
  const bool headOfTraj = i == 0; // head thread for own trajectory

  /* initialise */
  if (headOfTraj) {
    t = t1;
    h = h0;
    logfacold = bi::exp(BI_REAL(1.0e-4));
  }
  __syncthreads();

  /* integrate */
  real x0, x1, x2, x3, x4, x5, x6, err;
  real k1, k7;

  int n = 0;
  x0 = x;

  do {
    if (headOfTraj && t + BI_REAL(1.01)*h - t2 > BI_REAL(0.0)) {
      h = t2 - t;
    }
    __syncthreads();

    /* stages */
    Visitor::stage1(t, h, s, p, i, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
    k1in = true; // can reuse from previous iteration in future
    __syncthreads();
    x = x1;
    __syncthreads();

    Visitor::stage2(t, h, s, p, i, pax, x0, x2, x3, x4, x5, x6, err);
    __syncthreads();
    x = x2;
    __syncthreads();

    Visitor::stage3(t, h, s, p, i, pax, x0, x3, x4, x5, x6, err);
    __syncthreads();
    x = x3;
    __syncthreads();

    Visitor::stage4(t, h, s, p, i, pax, x0, x4, x5, x6, err);
    __syncthreads();
    x = x4;
    __syncthreads();

    Visitor::stage5(t, h, s, p, i, pax, x0, x5, x6, err);
    __syncthreads();
    x = x5;
    __syncthreads();

    Visitor::stage6(t, h, s, p, i, pax, x0, x6, err);
    __syncthreads();
    x = x6;
    __syncthreads();

    /* compute error */
    Visitor::stageErr(t, h, s, p, i, pax, x0, x6, k7, err);
    err *= h;
    err /= atoler + rtoler*bi::max(bi::abs(x0), bi::abs(x6));

    /* sum squared errors */
    /* have tried a spin lock here instead, using atomicCAS(), with slightly
     * worse performance */
    if (headOfTraj) {
      e2 = err*err;
    }
    __syncthreads();
    for (int j = 1; j < N; ++j) {
      if (i == j) {
        e2 += err*err;
      }
      __syncthreads();
    }

    /* accept or reject step */
    if (e2 <= BI_REAL(N)) {
      /* accept */
      k1 = k7;
      x0 = x6;
    } else {
      /* reject */
      x = x0;
      k1in = false;
    }

    /* handle step size */
    if (headOfTraj) {
      /* scale squared error */
      e2 *= BI_REAL(1.0)/BI_REAL(N);

      /* compute next step size */
      real logfac11 = expo*bi::log(e2);
      if (e2 > BI_REAL(1.0)) {
        /* step was rejected */
        h *= bi::max(facl, bi::exp(logsafe - logfac11));
      } else {
        /* step was accepted */
        t += h; // slightly faster to do this here, saves headOfTraj check
        h *= bi::min(facr, bi::max(facl, bi::exp(::beta*logfacold + logsafe - logfac11))); // bound
        logfacold = BI_REAL(0.5)*bi::log(bi::max(e2, BI_REAL(1.0e-8)));
      }
    }

    /* check if we're done */
    /* have tried with warp vote, slower */
    done = true;
    __syncthreads();
    if (headOfTraj && t < t2) {
      done = false;
    }
    __syncthreads();

    ++n;
  } while (!done && n < nsteps);
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_RK43KERNELGPU_CUH
#define BI_CUDA_ODE_RK43KERNELGPU_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for RK43IntegratorGPU.
 *
 * @tparam B Model type.
 *
 * @param t1 Current time.
 * @param t2 Time to which to integrate.
 * @param[in,out] s State.
 */
template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void kernelRK43(const T1 t1, const T1 t2,
    State<B,ON_DEVICE> s);

}

#include "RK43VisitorGPU.cuh"
#include "IntegratorConstants.cuh"
#include "../constant.cuh"
#include "../shared.cuh"
#include "../global.cuh"

template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void bi::kernelRK43(const T1 t1, const T1 t2, State<B,ON_DEVICE> s) {
  typedef Pa<ON_DEVICE,B,global,global,global,global> PX;
  typedef RK43VisitorGPU<B,S,S,real,PX,real> Visitor;

  /* sizes */
  const int P = s.size();
  const int N = block_size<S>::value;
  const int ND = B::ND;

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
  const bool headOfTraj = i == 0; // head thread for own trajectory

  /* initialise */
  if (headOfTraj) {
    t = t1;
    h = h0;
    logfacold = bi::log(BI_REAL(1.0e-4));
  }
  __syncthreads();

  /* integrate */
  real r1, r2, old, err;

  int n = 0;
  old = x;
  r1 = x;

  do {
    if (headOfTraj && t + BI_REAL(1.01)*h - t2 > BI_REAL(0.0)) {
      h = t2 - t;
    }
    __syncthreads();

    /* stages */
    Visitor::stage1(t, h, s, p, i, pax, r1, r2, err);
    __syncthreads();
    x = r1;
    __syncthreads();

    Visitor::stage2(t, h, s, p, i, pax, r1, r2, err);
    __syncthreads();
    x = r2;
    __syncthreads();

    Visitor::stage3(t, h, s, p, i, pax, r1, r2, err);
    __syncthreads();
    x = r1;
    __syncthreads();

    Visitor::stage4(t, h, s, p, i, pax, r1, r2, err);
    __syncthreads();
    x = r2;
    __syncthreads();

    Visitor::stage5(t, h, s, p, i, pax, r1, r2, err);
    __syncthreads();
    x = r1;
    __syncthreads();

    err *= h;
    err /= atoler + rtoler*bi::max(bi::abs(old), bi::abs(r1));

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
      old = r1;
    } else {
      /* reject */
      r1 = old;
      x = old;
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

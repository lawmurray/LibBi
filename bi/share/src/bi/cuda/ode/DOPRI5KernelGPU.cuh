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
 */
template<class B, class S, class T1>
CUDA_FUNC_GLOBAL void kernelDOPRI5(const T1 t1, const T1 t2);

}

#include "DOPRI5VisitorGPU.cuh"
#include "IntegratorConstants.cuh"
#include "../constant.cuh"
#include "../shared.cuh"
#include "../global.cuh"

template<class B, class S, class T1>
void bi::kernelDOPRI5(const T1 t1, const T1 t2) {
  typedef Pa<ON_DEVICE,B,real,constant,global,global,shared<S> > PX;
  typedef DOPRI5VisitorGPU<B,S,S,real,PX,real> Visitor;

  /* indices */
  const int i = threadIdx.y; // variable index
  const int p = blockIdx.x*blockDim.x + threadIdx.x; // trajectory index
  const int q = i*blockDim.x + threadIdx.x; // shared memory index
  const int r = threadIdx.x; // trajectory index in shared memory
  const int N = block_size<S>::value;

  /* shared memory */
  real* __restrict__ xs = shared_mem;
  real* __restrict__ ts = xs + N*blockDim.x;
  real* __restrict__ hs = ts + blockDim.x;
  real* __restrict__ e2s = hs + blockDim.x;
  real* __restrict__ logfacolds = e2s + blockDim.x;

  /* refs for this thread */
  real& x = xs[q];
  real& t = ts[r];
  real& h = hs[r];
  real& e2 = e2s[r];
  real& logfacold = logfacolds[r];
  PX pax;

  /* flags */
  bool k1in = false;
  const bool headOfTraj = i == 0; // head thread for own trajectory
  CUDA_VAR_SHARED bool done;

  /* initialise */
  if (headOfTraj) {
    t = t1;
    h = h0;
    logfacold = BI_MATH_LOG(BI_REAL(1.0e-4));
  }
  __syncthreads();

  /* integrate */
  real x0, x1, x2, x3, x4, x5, x6, err;
  real k1, k7;

  int n = 0;
  shared_init<B,S>(p, i);
  x0 = x;

  do {
    if (headOfTraj) {
      if (BI_REAL(0.1)*BI_MATH_FABS(h) <= BI_MATH_FABS(t)*uround) {
        // step size too small
      }
      if (t + BI_REAL(1.01)*h - t2 > BI_REAL(0.0)) {
        h = t2 - t;
      }
    }
    __syncthreads();

    /* stages */
    Visitor::stage1(t, h, p, i, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
    k1in = true; // can reuse from previous iteration in future
    __syncthreads();
    x = x1;
    __syncthreads();

    Visitor::stage2(t, h, p, i, pax, x0, x2, x3, x4, x5, x6, err);
    __syncthreads();
    x = x2;
    __syncthreads();

    Visitor::stage3(t, h, p, i, pax, x0, x3, x4, x5, x6, err);
    __syncthreads();
    x = x3;
    __syncthreads();

    Visitor::stage4(t, h, p, i, pax, x0, x4, x5, x6, err);
    __syncthreads();
    x = x4;
    __syncthreads();

    Visitor::stage5(t, h, p, i, pax, x0, x5, x6, err);
    __syncthreads();
    x = x5;
    __syncthreads();

    Visitor::stage6(t, h, p, i, pax, x0, x6, err);
    __syncthreads();
    x = x6;
    __syncthreads();

    /* compute error */
    Visitor::stageErr(t, h, p, i, pax, x0, x6, k7, err);
    err *= h;
    err /= atoler + rtoler*BI_MATH_MAX(BI_MATH_FABS(x0), BI_MATH_FABS(x6));

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
      real logfac11 = expo*BI_MATH_LOG(e2);
      if (e2 > BI_REAL(1.0)) {
        /* step was rejected */
        h *= BI_MATH_MAX(facl, BI_MATH_EXP(logsafe - logfac11));
      } else {
        /* step was accepted */
        t += h; // slightly faster to do this here, saves headOfTraj check
        h *= BI_MATH_MIN(facr, BI_MATH_MAX(facl, BI_MATH_EXP(beta*logfacold + logsafe - logfac11))); // bound
        logfacold = BI_REAL(0.5)*BI_MATH_LOG(BI_MATH_MAX(e2, BI_REAL(1.0e-8)));
      }
    }

    /* check if we're done */
    /* have tried with warp vote, slower */
    #ifdef ENABLE_RIPEN
    const int Q = blockDim.x*gridDim.x; // no. simultaneous trajectories
    #endif
    done = true;
    __syncthreads();
    if (t < t2) {
      if (headOfTraj) {
        done = false;
      }
    } else {
      #ifdef ENABLE_RIPEN
      if (p + Q < constP) {
        /* write result for this trajectory */
        shared_commit<B,S>(p, i);

        /* read starting state for next trajectory */
        shared_init<B,S>(p + Q, i);
        x = x0;
        k1in = false;
        n = 0;
      }
      #endif
    }
    __syncthreads();

    #ifdef ENABLE_RIPEN
    if (headOfTraj && t >= t2 && p + Q < constP) { // moved after __syncthreads() for pax.p update
      t = t1;
      h = h0;
      logfacold = BI_MATH_LOG(BI_REAL(1.0e-4));
      p += Q;
      done = false;
    }
    __syncthreads();
    #endif

    ++n;
  } while (!done && n < nsteps);

  /* commit back to global memory */
  shared_commit<B,S>(p, i);
}

#endif

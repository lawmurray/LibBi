/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_DOPRI5INTEGRATOR_CUH
#define BI_CUDA_ODE_DOPRI5INTEGRATOR_CUH

#include "DOPRI5Visitor.cuh"
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
void bi::DOPRI5Integrator<bi::ON_DEVICE,B,SH>::stepTo(const real tcur, const real tnxt) {
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,constant,global>::type pa;
  #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
  typedef Pa<B,real,pa,global,texture,pa,texture,shared,global> V1;
  #else
  typedef Pa<B,real,pa,global,global,pa,global,shared,global> V1;
  #endif
  typedef DOPRI5Visitor<ON_DEVICE,B,S,V1> Visitor;

  /* indices */
  const int id = threadIdx.y;
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int q = id*blockDim.x + threadIdx.x; // thread id
  const int r = threadIdx.x; // trajectory id in shared memory

  /* shared memory */
  real* __restrict__ xs = shared_mem;
  real* __restrict__ ts = xs + net_size<B,S>::value*blockDim.x;
  real* __restrict__ hs = ts + blockDim.x;
  real* __restrict__ e2s = hs + blockDim.x;
  real* __restrict__ logfacolds = e2s + blockDim.x;
  V1* __restrict__ paxs = (V1*)(logfacolds + blockDim.x);

  /* refs for this thread */
  real& x = xs[q];
  real& t = ts[r];
  real& h = hs[r];
  real& e2 = e2s[r];
  real& logfacold = logfacolds[r];
  V1& pax = paxs[r];

  /* flags */
  bool k1in = false;
  const bool headOfTraj = id == 0; // head thread for own trajectory
  CUDA_VAR_SHARED bool done;

  /* initialise */
  if (headOfTraj) {
    t = tcur;
    h = h0;
    logfacold = CUDA_LOG(REAL(1.0e-4));
    pax = V1(p);
  }
  __syncthreads();

  /* integrate */
  real x0, x1, x2, x3, x4, x5, x6, err;
  real k1, k7;

  int n = 0;
  #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
  x0 = tex2D(texCState, pax.p, id);
  #else
  x0 = globalCState(pax.p, id);
  #endif
  x = x0;

  do {
    if (headOfTraj) {
      if (REAL(0.1)*CUDA_ABS(h) <= CUDA_ABS(t)*uround) {
        // step size too small
      }
      if (t + REAL(1.01)*h - tnxt > REAL(0.0)) {
        h = tnxt - t;
      }
    }
    __syncthreads();

    /* stages */
    Visitor::stage1(t, h, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
    k1in = true; // can reuse from previous iteration in future
    __syncthreads();
    x = x1;
    __syncthreads();

    Visitor::stage2(t, h, pax, x0, x2, x3, x4, x5, x6, err);
    __syncthreads();
    x = x2;
    __syncthreads();

    Visitor::stage3(t, h, pax, x0, x3, x4, x5, x6, err);
    __syncthreads();
    x = x3;
    __syncthreads();

    Visitor::stage4(t, h, pax, x0, x4, x5, x6, err);
    __syncthreads();
    x = x4;
    __syncthreads();

    Visitor::stage5(t, h, pax, x0, x5, x6, err);
    __syncthreads();
    x = x5;
    __syncthreads();

    Visitor::stage6(t, h, pax, x0, x6, err);
    __syncthreads();
    x = x6;
    __syncthreads();

    /* compute error */
    Visitor::stageErr(t, h, pax, x0, x6, k7, err);
    err *= h;
    err /= atoler + rtoler*CUDA_MAX(CUDA_ABS(x0), CUDA_ABS(x6));

    /* sum squared errors */
    /* have tried a spin lock here instead, using atomicCAS(), with slightly
     * worse performance */
    if (headOfTraj) {
      e2 = err*err;
    }
    __syncthreads();
    for (int i = 1; i < net_size<B,S>::value; i++) {
      if (id == i) {
        e2 += err*err;
      }
      __syncthreads();
    }

    /* accept or reject step */
    if (e2 <= REAL(net_size<B,S>::value)) {
      /* accept */
      k1 = k7;
      x0 = x6;
    } else {
      /* reject */
      x = x0;
    }

    /* handle step size */
    if (headOfTraj) {
      /* scale squared error */
      e2 *= REAL(1.0) / REAL(net_size<B,S>::value);

      /* compute next step size */
      real logfac11 = expo*CUDA_LOG(e2);
      if (e2 > REAL(1.0)) {
        /* step was rejected */
        h *= CUDA_MAX(facl, CUDA_EXP(logsafe - logfac11));
      } else {
        /* step was accepted */
        t += h; // slightly faster to do this here, saves headOfTraj check
        h *= CUDA_MIN(facr, CUDA_MAX(facl, CUDA_EXP(beta*logfacold + logsafe - logfac11))); // bound
        logfacold = REAL(0.5)*CUDA_LOG(CUDA_MAX(e2, REAL(1.0e-8)));
      }
    }

    /* check if we're done */
    /* have tried with warp vote, slower */
    #ifdef USE_RIPEN
    const int Q = blockDim.x*gridDim.x; // no. simultaneous trajectories
    #endif
    done = true;
    __syncthreads();
    if (t < tnxt) {
      if (headOfTraj) {
        done = false;
      }
    } else {
      #ifdef USE_RIPEN
      if (pax.p + Q < constP) {
        /* write result for this trajectory */
        globalCState(pax.p, id) = x;

        /* read starting state for next trajectory */
        #if !defined(USE_DOUBLE) and defined(USE_TEXTURE)
        x0 = tex2D(texCState, pax.p + Q, id);
        #else
        x0 = globalCState(pax.p + Q, id);
        #endif
        x = x0;
        k1in = false;
        n = 0;
      }
      #endif
    }
    __syncthreads();

    #ifdef USE_RIPEN
    if (headOfTraj && t >= tnxt && pax.p + Q < constP) { // moved after __syncthreads() for pax.p update
      t = tcur;
      h = h0;
      logfacold = CUDA_LOG(REAL(1.0e-4));
      pax.p += Q;
      done = false;
    }
    __syncthreads();
    #endif

    ++n;
  } while (!done && n < nsteps);

  /* write result for this trajectory */
  globalCState(pax.p, id) = x;
}

#endif

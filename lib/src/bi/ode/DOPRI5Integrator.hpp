/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_DOPRI5INTEGRATOR_HPP
#define BI_ODE_DOPRI5INTEGRATOR_HPP

namespace bi {
/**
 * DOPRI5(4) integrator.
 *
 * @ingroup method_updater
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam SH StaticHandling type. Denoted unsigned for compatibility of CUDA
 * device code.
 *
 * Implements the DOPRI5(4) method as described in @ref Hairer1993
 * "Hairer \& Norsett (1993)". Adapted from IntegratorT by Blake Ashby
 * <bmashby@stanford.edu>. Implementation described in @ref Murray2011
 * "Murray (2011)".
 */
template<Location L, class B, unsigned SH = STATIC_SHARED>
class DOPRI5Integrator {
public:
  /**
   * Integrate forward.
   *
   * @param tcur Current time.
   * @param tnxt Time to which to integrate.
   */
  static void stepTo(const real tcur, const real tnxt);
};

/**
 * @internal
 *
 * Host implementation of DOPRI5Integrator.
 */
template<class B, unsigned SH>
class DOPRI5Integrator<ON_HOST,B,SH> {
public:
  static void stepTo(const real tcur, const real tnxt);
};

/**
 * @internal
 *
 * Device implementation of DOPRI5Integrator.
 */
template<class B, unsigned SH>
class DOPRI5Integrator<ON_DEVICE,B,SH> {
public:
  static CUDA_FUNC_DEVICE void stepTo(const real tcur, const real tnxt);
};

}

#include "DOPRI5Visitor.hpp"
#include "IntegratorConstants.hpp"
#include "../host/host.hpp"
#include "../state/Pa.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"

#ifdef USE_SSE
#include "../math/sse_state.hpp"
#include "../host/sse_host.hpp"
#include "../host/sse_const_host.hpp"
#endif

template<class B, unsigned SH>
void bi::DOPRI5Integrator<bi::ON_HOST,B,SH>::stepTo(const real tcur, const real tnxt) {
  /* pre-condition */
  assert (tcur < tnxt);

  #ifdef USE_SSE
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,sse_const_host,sse_host>::type pa;
  typedef Pa<B,sse_real,pa,sse_host,sse_host,pa,sse_host,sse_host,sse_host> V1;
  typedef DOPRI5Visitor<ON_HOST,B,S,sse_real,V1> Visitor;
  static const int N = net_size<B,S>::value;
  const int P = hostCState.size1();

  #pragma omp parallel
  {
    bool k1in;
    int id, n, p;

    sse_real x0[N], x1[N], x2[N], x3[N], x4[N], x5[N], x6[N], err[N], k1[N], k7[N], t, h, e2, logfacold, logfac11, fac;
    V1 pax(0);

    #pragma omp for
    for (p = 0; p < P; p += BI_SSE_SIZE) {
      pax.p = p;
      t = tcur;
      h = h_h0;
      logfacold = CUDA_LOG(REAL(1.0e-4));
      k1in = false;
      n = 0;
      for (id = 0; id < N; ++id) {
        x0[id] = sse_state_get(hostCState, p, id);
      }

      /* integrate */
      sse_real cond, accept;
      bool done = false;
      while (!done && n < h_nsteps) {
        cond = sse_gt(t + h*REAL(1.01) - tnxt, REAL(0.0));
        h = sse_if(cond, tnxt - t, h);
        //t = sse_if(sse_le(h, REAL(0.0)), tnxt, t);

        /* stages */
        Visitor::stage1(t, h, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
        k1in = true; // can reuse from previous iteration in future
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x1[id]);
        }

        Visitor::stage2(t, h, pax, x0, x2, x3, x4, x5, x6, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x2[id]);
        }

        Visitor::stage3(t, h, pax, x0, x3, x4, x5, x6, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x3[id]);
        }

        Visitor::stage4(t, h, pax, x0, x4, x5, x6, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x4[id]);
        }

        Visitor::stage5(t, h, pax, x0, x5, x6, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x5[id]);
        }

        Visitor::stage6(t, h, pax, x0, x6, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x6[id]);
        }

        /* compute error */
        Visitor::stageErr(t, h, pax, x0, x6, k7, err);

        e2 = REAL(0.0);
        for (id = 0; id < N; ++id) {
          err[id] *= h;
          err[id] /= sse_max(sse_fabs(x0[id]), sse_fabs(x6[id]))*h_rtoler + h_atoler;
          e2 += err[id]*err[id];
        }
        e2 *= REAL(1.0) / REAL(N);

        /* compute next step size and accept/reject */
        accept = sse_le(e2, REAL(1.0)); // accepted?
        t = sse_if(accept, t + h, t);
        for (id = 0; id < N; ++id) {
          k1[id] = sse_if(accept, k7[id], k1[id]);
          x0[id] = sse_if(accept, x6[id], x0[id]);
          sse_state_set(hostCState, p, id, x0[id]);
        }

        logfac11 = sse_log(e2)*h_expo;
        fac = sse_exp(logfacold*h_beta + h_logsafe - logfac11); // Lund-stabilization
        fac = sse_min(h_facr, sse_max(h_facl, fac)); // bound
        h *= sse_if(accept, fac, sse_max(h_facl, sse_exp(h_logsafe - logfac11)));
        logfacold = sse_if(accept, REAL(0.5)*sse_log(sse_max(e2, REAL(1.0e-8))), logfacold);

        /* determine if we're done */
        done = !sse_any(sse_lt(t, tnxt));
        ++n;
      }
    }
  }
  #else
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,const_host,host>::type pa;
  typedef Pa<B,real,pa,host,host,pa,host,host,host> V1;
  typedef DOPRI5Visitor<ON_HOST,B,S,real,V1> Visitor;
  static const int N = net_size<B,S>::value;
  const int P = hostCState.size1();

  #pragma omp parallel
  {
    bool k1in;
    int id, n, p;

    real x0[N], x1[N], x2[N], x3[N], x4[N], x5[N], x6[N], err[N], k1[N],
        k7[N], t, h, e2, logfacold, logfac11, fac;
    V1 pax(0);

    #pragma omp for
    for (p = 0; p < P; ++p) {
      pax.p = p;
      t = tcur;
      h = h_h0;
      logfacold = CUDA_LOG(REAL(1.0e-4));
      k1in = false;
      n = 0;

      for (id = 0; id < N; ++id) {
        x0[id] = hostCState(p, id);
      }

      /* integrate */
      while (t < tnxt && n < h_nsteps) {
        if (REAL(0.1)*CUDA_ABS(h) <= CUDA_ABS(t)*h_uround) {
          // step size too small
        }
        if (t + REAL(1.01)*h - tnxt > REAL(0.0)) {
          h = tnxt - t;
          if (h <= 0.0) {
            t = tnxt;
            break;
          }
        }

        /* stages */
        Visitor::stage1(t, h, pax, x0, x1, x2, x3, x4, x5, x6, k1, err, k1in);
        k1in = true; // can reuse from previous iteration in future
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x1[id];
        }

        Visitor::stage2(t, h, pax, x0, x2, x3, x4, x5, x6, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x2[id];
        }

        Visitor::stage3(t, h, pax, x0, x3, x4, x5, x6, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x3[id];
        }

        Visitor::stage4(t, h, pax, x0, x4, x5, x6, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x4[id];
        }

        Visitor::stage5(t, h, pax, x0, x5, x6, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x5[id];
        }

        Visitor::stage6(t, h, pax, x0, x6, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x6[id];
        }

        /* compute error */
        Visitor::stageErr(t, h, pax, x0, x6, k7, err);

        e2 = 0.0;
        for (id = 0; id < N; ++id) {
          err[id] *= h;
          err[id] /= h_atoler + h_rtoler*CUDA_MAX(CUDA_ABS(x0[id]), CUDA_ABS(x6[id]));
          e2 += err[id]*err[id];
        }
        e2 *= REAL(1.0) / REAL(N);

        /* accept/reject */
        if (e2 <= REAL(1.0)) {
          /* accept */
          t += h;
          for (id = 0; id < N; ++id) {
            x0[id] = x6[id];
            k1[id] = k7[id];
            hostCState(pax.p, id) = x6[id];
          }
        }

        /* compute next step size */
        logfac11 = h_expo*CUDA_LOG(e2);
        if (e2 > REAL(1.0)) {
          /* step was rejected */
          h *= CUDA_MAX(h_facl, CUDA_EXP(h_logsafe - logfac11));
        } else {
          /* step was accepted */
          fac = CUDA_EXP(h_beta*logfacold + h_logsafe - logfac11); // Lund-stabilization
          fac = CUDA_MIN(h_facr, CUDA_MAX(h_facl, fac)); // bound
          h *= fac;
          logfacold = REAL(0.5)*CUDA_LOG(CUDA_MAX(e2, REAL(1.0e-8)));
        }

        ++n;
      }
    }
  }
  #endif
}

#endif

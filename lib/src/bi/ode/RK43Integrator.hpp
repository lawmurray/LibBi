/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_RK43INTEGRATOR_HPP
#define BI_ODE_RK43INTEGRATOR_HPP

namespace bi {
/**
 * @internal
 *
 * RK4(3)5[2R+]C integrator.
 *
 * @ingroup method_updater
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam SH StaticHandling type. Denoted unsigned for compatibility of CUDA
 * device code.
 *
 *
 * Implements the RK4(3)5[2R+]C method as described in @ref Kennedy2000
 * "Kennedy et. al. (2000)". Implementation described in @ref Murray2011
 * "Murray (2011)".
 *
 */
template<Location L, class B, unsigned SH = STATIC_SHARED>
class RK43Integrator {
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
 * Host implementation of RK43Integrator.
 */
template<class B, unsigned SH>
class RK43Integrator<ON_HOST,B,SH> {
public:
  static void stepTo(const real tcur, const real tnxt);
};

/**
 * @internal
 *
 * Device implementation of RK43Integrator.
 */
template<class B, unsigned SH>
class RK43Integrator<ON_DEVICE,B,SH> {
public:
  static CUDA_FUNC_DEVICE void stepTo(const real tcur, const real tnxt);
};
}

#include "RK43Visitor.hpp"
#include "IntegratorConstants.hpp"
#include "../host/host.hpp"
#include "../state/Pa.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../math/view.hpp"

#ifdef USE_SSE
#include "../math/sse_state.hpp"
#include "../host/sse_host.hpp"
#include "../host/sse_const_host.hpp"
#endif

template<class B, unsigned SH>
void bi::RK43Integrator<bi::ON_HOST,B,SH>::stepTo(const real tcur, const real tnxt) {
  #ifdef USE_SSE
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,sse_const_host,sse_host>::type pa;
  typedef Pa<B,sse_real,pa,sse_host,sse_host,pa,sse_host,sse_host,sse_host> V1;
  typedef RK43Visitor<ON_HOST,B,S,V1> Visitor;
  static const int N = net_size<B,S>::value;
  const int P = hostCState.size1();

  #pragma omp parallel
  {
    int id, n, p;
    sse_real r1[N], r2[N], err[N], old[N], t, h, e2, logfacold, logfac11, fac;
    V1 pax(0);

    #pragma omp for
    for (p = 0; p < P; p += BI_SSE_SIZE) {
      pax.p = p;
      t = tcur;
      h = h_h0;
      logfacold = CUDA_LOG(REAL(1.0e-4));
      n = 0;
      for (id = 0; id < N; ++id) {
        old[id] = sse_state_get(hostCState, p, id);
        r1[id] = old[id];
      }

      /* integrate */
      sse_real cond, accept;
      bool done = false;
      while (!done && n < h_nsteps) {
        cond = sse_gt(t + h*REAL(1.01) - tnxt, REAL(0.0));
        h = sse_if(cond, tnxt - t, h);
        //t = sse_if(sse_le(h, REAL(0.0)), tnxt, t);

        /* stages */
        Visitor::stage1(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, r1[id]);
        }

        Visitor::stage2(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, r2[id]);
        }

        Visitor::stage3(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, r1[id]);
        }

        Visitor::stage4(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, r2[id]);
        }

        Visitor::stage5(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, r1[id]);
        }

        e2 = REAL(0.0);
        for (id = 0; id < N; ++id) {
          err[id] *= h;
          err[id] /= sse_max(sse_fabs(old[id]), sse_fabs(r1[id]))*h_rtoler + h_atoler;
          e2 += err[id]*err[id];
        }
        e2 *= REAL(1.0) / REAL(N);

        /* compute next step size and accept/reject */
        accept = sse_le(e2, REAL(1.0)); // accepted?
        t = sse_if(accept, t + h, t);
        for (id = 0; id < N; ++id) {
          old[id] = sse_if(accept, r1[id], old[id]); // if accepted
          r1[id] = old[id];
          sse_state_set(hostCState, p, id, old[id]);
        }

        logfac11 = sse_log(e2)*h_expo;
        fac = sse_exp(logfacold*h_beta + h_logsafe - logfac11); // Lund-stabilization
        fac = sse_min(h_facr, sse_max(h_facl, fac)); // bound
        h *= sse_if(accept, fac, sse_max(h_facl, sse_exp(h_logsafe - logfac11)));
        logfacold = sse_if(accept, REAL(0.5)*sse_log(sse_max(e2, REAL(1.0e-8))), logfacold);

//        fac11 = sse_pow(e2, h_expo);
//        fac = fac11 / sse_pow(facold, h_beta) / h_safe;
//        fac = sse_max(h_facc2, sse_min(h_facc1, fac)); // bound
//        h /= sse_if(accept, sse_min(h_facc1, fac11 / h_safe), fac);
//        facold = sse_if(accept, facold, facold*sse_sqrt(sse_max(e2, 1.0e-8)));

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
  typedef RK43Visitor<ON_HOST,B,S,V1> Visitor;
  static const int N = net_size<B,S>::value;
  const int P = hostCState.size1();

  #pragma omp parallel
  {
    int id, n, p;
    real r1[N], r2[N], err[N], old[N], t, h, e2, logfacold, logfac11, fac;
    V1 pax(0);

    #pragma omp for
    for (p = 0; p < P; ++p) {
      pax.p = p;
      t = tcur;
      h = h_h0;
      logfacold = CUDA_LOG(REAL(1.0e-4));
      n = 0;
      for (id = 0; id < N; ++id) {
        old[id] = hostCState(pax.p, id);
        r1[id] = old[id];
      }

      /* integrate */
      while (t < tnxt && n < h_nsteps) {
        if (REAL(0.1)*CUDA_ABS(h) <= CUDA_ABS(t)*h_uround) {
          // step size too small
        }
        if (t + REAL(1.01)*h - tnxt > REAL(0.0)) {
          h = tnxt - t;
          if (h <= REAL(0.0)) {
            t = tnxt;
            break;
          }
        }

        /* stages */
        Visitor::stage1(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = r1[id];
        }

        Visitor::stage2(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = r2[id];
        }

        Visitor::stage3(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = r1[id];
        }

        Visitor::stage4(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = r2[id];
        }

        Visitor::stage5(t, h, pax, r1, r2, err);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = r1[id];
        }

        e2 = REAL(0.0);
        for (id = 0; id < N; ++id) {
          err[id] *= h;
          err[id] /= h_atoler + h_rtoler*CUDA_MAX(CUDA_ABS(old[id]), CUDA_ABS(r1[id]));
          e2 += err[id]*err[id];
        }
        e2 *= REAL(1.0) / REAL(N);

        if (e2 <= REAL(1.0)) {
          /* accept */
          t += h;
          for (id = 0; id < N; ++id) {
            old[id] = r1[id];
          }
        } else {
          /* reject */
          for (id = 0; id < N; ++id) {
            r1[id] = old[id];
          }
        }
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = old[id];
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

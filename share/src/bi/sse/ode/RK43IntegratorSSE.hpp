/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_ODE_RK43INTEGRATORSSE_HPP
#define BI_SSE_ODE_RK43INTEGRATORSSE_HPP

namespace bi {
/**
 * @copydoc RK43Integrator
 */
template<class B, class S, class T1>
class RK43IntegratorSSE {
public:
  /**
   * @copydoc RK43Integrator::integrate()
   */
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);
};
}

#include "../sse_host.hpp"
#include "../../host/ode/RK43VisitorHost.hpp"
#include "../../host/ode/IntegratorConstants.hpp"
#include "../../state/Pa.hpp"
#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class T1>
void bi::RK43IntegratorSSE<B,S,T1>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-condition */
  BI_ASSERT(t1 < t2);

  typedef typename temp_host_vector<simd_real>::type vector_type;
  typedef Pa<ON_HOST,B,host,host,sse_host,sse_host> PX;
  typedef RK43VisitorHost<B,S,S,real,PX,simd_real> Visitor;
  static const int N = block_size<S>::value;
  const int P = s.size();

  #pragma omp parallel
  {
    vector_type r1(N), r2(N), err(N), old(N);
    simd_real e, e2;
    real t, h, logfacold, logfac11, fac, e2max;
    int n, id, p;
    PX pax;

    #pragma omp for
    for (p = 0; p < P; p += BI_SIMD_SIZE) {
      t = t1;
      h = h_h0;
      logfacold = bi::log(BI_REAL(1.0e-4));
      n = 0;
      sse_host_load<B,S>(s, p, old);
      r1 = old;

      /* integrate */
      while (t < t2 && n < h_nsteps) {
        if (BI_REAL(0.1)*bi::abs(h) <= bi::abs(t)*h_uround) {
          // step size too small
        }
        if (t + BI_REAL(1.01)*h - t2 > BI_REAL(0.0)) {
          h = t2 - t;
          if (h <= BI_REAL(0.0)) {
            t = t2;
            break;
          }
        }

        /* stages */
        Visitor::stage1(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        sse_host_store<B,S>(s, p, r1);

        Visitor::stage2(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        sse_host_store<B,S>(s, p, r2);

        Visitor::stage3(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        sse_host_store<B,S>(s, p, r1);

        Visitor::stage4(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        sse_host_store<B,S>(s, p, r2);

        Visitor::stage5(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        sse_host_store<B,S>(s, p, r1);

        /* determine largest error among trajectories */
        e2 = BI_REAL(0.0);
        for (id = 0; id < N; ++id) {
          e = err(id)*h/(bi::max(bi::abs(old(id)), bi::abs(r1(id)))*h_rtoler + h_atoler);
          e2 += e*e;
        }

        e2max = bi::max_reduce(e2)/N;
        if (e2max <= BI_REAL(1.0)) {
          /* accept */
          t += h;
          if (t < t2) {
            old = r1;
          }
        } else {
          /* reject */
          r1 = old;
          sse_host_store<B,S>(s, p, old);
        }

        /* compute next step size */
        if (t < t2) {
          logfac11 = h_expo*bi::log(e2max);
          if (e2max > BI_REAL(1.0)) {
            /* step was rejected */
            h *= bi::max(h_facl, bi::exp(h_logsafe - logfac11));
          } else {
            /* step was accepted */
            fac = bi::exp(h_beta*logfacold + h_logsafe - logfac11); // Lund-stabilization
            fac = bi::min(h_facr, bi::max(h_facl, fac)); // bound
            h *= fac;
            logfacold = BI_REAL(0.5)*bi::log(bi::max(e2max, BI_REAL(1.0e-8)));
          }
        }

        ++n;
      }
    }
  }
}

#endif

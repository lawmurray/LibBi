/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_ODE_RK43INTEGRATORHOST_HPP
#define BI_HOST_ODE_RK43INTEGRATORHOST_HPP

namespace bi {
/**
 * RK4(3)5[2R+]C low-storage Runge-Kutta integrator.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 *
 * Implements the RK4(3)5[2R+]C method as described in @ref Kennedy2000
 * "Kennedy et. al. (2000)". Implementation described in @ref Murray2011
 * "Murray (2011)".
 */
template<class B, class S, class T1>
class RK43IntegratorHost {
public:
  /**
   * Integrate.
   *
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param[in,out] s State.
   */
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);
};
}

#include "RK43VisitorHost.hpp"
#include "IntegratorConstants.hpp"
#include "../host.hpp"
#include "../../state/Pa.hpp"
#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/block_traits.hpp"
#include "../../math/view.hpp"
#include "../../math/temp_vector.hpp"

template<class B, class S, class T1>
void bi::RK43IntegratorHost<B,S,T1>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-condition */
  BI_ASSERT(t1 < t2);

  typedef typename temp_host_vector<real>::type vector_type;
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef RK43VisitorHost<B,S,S,real,PX,real> Visitor;

  static const int N = block_size<S>::value;
  const int P = s.size();

  #pragma omp parallel
  {
    vector_type r1(N), r2(N), err(N), old(N);
    real t, h, e, e2, logfacold, logfac11, fac;
    int n, id, p;
    PX pax;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      t = t1;
      h = h_h0;
      logfacold = bi::log(BI_REAL(1.0e-4));
      n = 0;
      host_load<B,S>(s, p, old);
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
        host_store<B,S>(s, p, r1);

        Visitor::stage2(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        host_store<B,S>(s, p, r2);

        Visitor::stage3(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        host_store<B,S>(s, p, r1);

        Visitor::stage4(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        host_store<B,S>(s, p, r2);

        Visitor::stage5(t, h, s, p, pax, r1.buf(), r2.buf(), err.buf());
        host_store<B,S>(s, p, r1);

        /* compute error */
        e2 = BI_REAL(0.0);
        for (id = 0; id < N; ++id) {
          e = err(id)*h/(h_atoler + h_rtoler*bi::max(bi::abs(old(id)), bi::abs(r1(id))));
          e2 += e*e;
        }
        e2 /= N;

        if (e2 <= BI_REAL(1.0)) {
          /* accept */
          t += h;
          if (t < t2) {
            old = r1;
          }
        } else {
          /* reject */
          r1 = old;
          host_store<B,S>(s, p, old);
        }

        /* compute next step size */
        if (t < t2) {
          logfac11 = h_expo*bi::log(e2);
          if (e2 > BI_REAL(1.0)) {
            /* step was rejected */
            h *= bi::max(h_facl, bi::exp(h_logsafe - logfac11));
          } else {
            /* step was accepted */
            fac = bi::exp(h_beta*logfacold + h_logsafe - logfac11); // Lund-stabilization
            fac = bi::min(h_facr, bi::max(h_facl, fac)); // bound
            h *= fac;
            logfacold = BI_REAL(0.5)*bi::log(bi::max(e2, BI_REAL(1.0e-8)));
          }
        }

        ++n;
      }
    }
  }
}

#endif

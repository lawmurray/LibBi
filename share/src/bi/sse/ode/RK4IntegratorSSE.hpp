/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_ODE_RK4INTEGRATORSSE_HPP
#define BI_SSE_ODE_RK4INTEGRATORSSE_HPP

namespace bi {
/**
 * @copydoc RK4Integrator
 */
template<class B, class S, class T1>
class RK4IntegratorSSE {
public:
  /**
   * @copydoc RK4Integrator::integrate()
   */
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);
};
}

#include "../sse_host.hpp"
#include "../../host/ode/RK4VisitorHost.hpp"
#include "../../host/ode/IntegratorConstants.hpp"
#include "../../state/Pa.hpp"
#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class T1>
void bi::RK4IntegratorSSE<B,S,T1>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-condition */
  BI_ASSERT(t1 < t2);

  typedef typename temp_host_vector<simd_real>::type vector_type;
  typedef Pa<ON_HOST,B,host,host,sse_host,sse_host> PX;
  typedef RK4VisitorHost<B,S,S,real,PX,simd_real> Visitor;
  static const int N = block_size<S>::value;
  const int P = s.size();

  #pragma omp parallel
  {
    vector_type x0(N), x1(N), x2(N), x3(N), x4(N);
    real t, h;
    int p;
    PX pax;

    #pragma omp for
    for (p = 0; p < P; p += BI_SIMD_SIZE) {
      t = t1;
      h = h_h0;

      /* integrate */
      while (t < t2) {
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
        sse_host_load<B,S>(s, p, x0);

        /* stages */
        Visitor::stage1(t, h, s, p, pax, x0.buf(), x1.buf(), x2.buf(), x3.buf(), x4.buf());
        sse_host_store<B,S>(s, p, x1);

        Visitor::stage2(t, h, s, p, pax, x0.buf(), x2.buf(), x3.buf(), x4.buf());
        sse_host_store<B,S>(s, p, x2);

        Visitor::stage3(t, h, s, p, pax, x0.buf(), x3.buf(), x4.buf());
        sse_host_store<B,S>(s, p, x3);

        Visitor::stage4(t, h, s, p, pax, x0.buf(), x4.buf());
        sse_host_store<B,S>(s, p, x4);

        t += h;
      }
    }
  }
}

#endif

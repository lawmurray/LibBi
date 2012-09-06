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
#include "../sse_shared_host.hpp"
#include "../math/function.hpp"
#include "../math/control.hpp"
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

  typedef host_vector_reference<sse_real> vector_reference_type;
  typedef Pa<ON_HOST,B,host,host,sse_host,sse_shared_host<S> > PX;
  typedef RK4VisitorHost<B,S,S,real,PX,sse_real> Visitor;
  static const int N = block_size<S>::value;
  const int P = s.size();

  bind(s);

  #pragma omp parallel
  {
    sse_real buf[6*N]; // use of dynamic array faster than heap allocation
    vector_reference_type x0(buf, N);
    vector_reference_type x1(buf + N, N);
    vector_reference_type x2(buf + 2*N, N);
    vector_reference_type x3(buf + 3*N, N);
    vector_reference_type x4(buf + 4*N, N);
    vector_reference_type shared(buf + 5*N, N);
    sseSharedHostState = &shared;

    real t, h;
    int p;
    PX pax;

    #pragma omp for
    for (p = 0; p < P; p += BI_SSE_SIZE) {
      /* initialise shared memory from global memory */
      sse_shared_host_init<B,S>(s, p);

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
        x0 = *sseSharedHostState;

        /* stages */
        Visitor::stage1(t, h, s, p, pax, x0.buf(), x1.buf(), x2.buf(), x3.buf(), x4.buf());
        sseSharedHostState->swap(x1);

        Visitor::stage2(t, h, s, p, pax, x0.buf(), x2.buf(), x3.buf(), x4.buf());
        sseSharedHostState->swap(x2);

        Visitor::stage3(t, h, s, p, pax, x0.buf(), x3.buf(), x4.buf());
        sseSharedHostState->swap(x3);

        Visitor::stage4(t, h, s, p, pax, x0.buf(), x4.buf());
        sseSharedHostState->swap(x4);

        t += h;
      }

      /* write from shared back to global memory */
      sse_shared_host_commit<B,S>(s, p);
    }
  }

  unbind(s);
}

#endif

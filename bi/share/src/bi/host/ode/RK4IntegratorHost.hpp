/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_ODE_RK4INTEGRATORHOST_HPP
#define BI_HOST_ODE_RK4INTEGRATORHOST_HPP

namespace bi {
/**
 * Classic fourth order Runge-Kutta integrator.
 *
 * @ingroup math_ode
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 */
template<class B, class S, class T1>
class RK4IntegratorHost {
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

#include "RK4VisitorHost.hpp"
#include "IntegratorConstants.hpp"
#include "../host.hpp"
#include "../shared_host.hpp"
#include "../../state/Pa.hpp"
#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/block_traits.hpp"
#include "../../math/view.hpp"

template<class B, class S, class T1>
void bi::RK4IntegratorHost<B,S,T1>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-condition */
  BI_ASSERT(t1 < t2);

  typedef host_vector_reference<real> vector_reference_type;
  typedef Pa<ON_HOST,B,host,host,host,shared_host<S> > PX;
  typedef RK4VisitorHost<B,S,S,real,PX,real> Visitor;

  static const int N = block_size<S>::value;
  const int P = s.size();

  bind(s);

  #pragma omp parallel
  {
    real buf[6*N]; // use of dynamic array faster than heap allocation
    vector_reference_type x0(buf, N);
    vector_reference_type x1(buf + N, N);
    vector_reference_type x2(buf + 2*N, N);
    vector_reference_type x3(buf + 3*N, N);
    vector_reference_type x4(buf + 4*N, N);
    vector_reference_type shared(buf + 5*N, N);
    sharedHostState = &shared;

    real t, h;
    int p;
    PX pax;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      /* initialise shared memory from global memory */
      shared_host_init<B,S>(s, p);

      t = t1;
      h = h_h0;

      /* integrate */
      while (t < t2) {
        /* initialise */
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
        x0 = *sharedHostState;

        /* stages */
        Visitor::stage1(t, h, s, p, pax, x0.buf(), x1.buf(), x2.buf(), x3.buf(), x4.buf());
        sharedHostState->swap(x1);

        Visitor::stage2(t, h, s, p, pax, x0.buf(), x2.buf(), x3.buf(), x4.buf());
        sharedHostState->swap(x2);

        Visitor::stage3(t, h, s, p, pax, x0.buf(), x3.buf(), x4.buf());
        sharedHostState->swap(x3);

        Visitor::stage4(t, h, s, p, pax, x0.buf(), x4.buf());
        sharedHostState->swap(x4);

        t += h;
      }

      /* write from shared back to global memory */
      shared_host_commit<B,S>(s, p);
    }
  }

  unbind(s);
}

#endif

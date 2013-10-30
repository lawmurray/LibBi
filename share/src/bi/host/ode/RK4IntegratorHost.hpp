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
 * @ingroup method_updater
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

  typedef typename temp_host_vector<real>::type vector_type;
  typedef Pa<ON_HOST,B,host,host,host,host> PX;
  typedef RK4VisitorHost<B,S,S,real,PX,real> Visitor;

  static const int N = block_size<S>::value;
  const int P = s.size();

  #pragma omp parallel
  {
    vector_type x0(N), x1(N), x2(N), x3(N), x4(N);
    real t, h;
    int p;
    PX pax;

    #pragma omp for
    for (p = 0; p < P; ++p) {
      t = t1;
      h = h_h0;
      host_load<B,S>(s, p, x0);

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

        /* stages */
        Visitor::stage1(t, h, s, p, pax, x0.buf(), x1.buf(), x2.buf(), x3.buf(), x4.buf());
        host_store<B,S>(s, p, x1);

        Visitor::stage2(t, h, s, p, pax, x0.buf(), x2.buf(), x3.buf(), x4.buf());
        host_store<B,S>(s, p, x2);

        Visitor::stage3(t, h, s, p, pax, x0.buf(), x3.buf(), x4.buf());
        host_store<B,S>(s, p, x3);

        Visitor::stage4(t, h, s, p, pax, x0.buf(), x4.buf());
        host_store<B,S>(s, p, x4);

        x0.swap(x4);
        t += h;
      }
    }
  }
}

#endif

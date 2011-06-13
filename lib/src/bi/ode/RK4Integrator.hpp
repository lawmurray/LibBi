/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1292 $
 * $Date: 2011-02-22 13:45:22 +0800 (Tue, 22 Feb 2011) $
 */
#ifndef BI_ODE_RK4INTEGRATOR_HPP
#define BI_ODE_RK4INTEGRATOR_HPP

namespace bi {
/**
 * RK4 (classic fourth-order Runge-Kutta) integrator.
 *
 * @ingroup method_updater
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam SH StaticHandling type. Denoted unsigned for compatibility of CUDA
 * device code.
 */
template<Location L, class B, unsigned SH = STATIC_SHARED>
class RK4Integrator {
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
 * Host implementation of RK4Integrator.
 */
template<class B, unsigned SH>
class RK4Integrator<ON_HOST,B,SH> {
public:
  static void stepTo(const real tcur, const real tnxt);
};

/**
 * @internal
 *
 * Device implementation of RK4Integrator.
 */
template<class B, unsigned SH>
class RK4Integrator<ON_DEVICE,B,SH> {
public:
  static CUDA_FUNC_DEVICE void stepTo(const real tcur, const real tnxt);
};

}

#include "RK4Visitor.hpp"
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
void bi::RK4Integrator<bi::ON_HOST,B,SH>::stepTo(const real tcur, const real tnxt) {
  /* pre-condition */
  assert (std::abs(tnxt - tcur) > 0.0);

  #ifdef USE_SSE
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,sse_const_host,sse_host>::type pa;
  typedef Pa<ON_HOST,B,sse_real,pa,sse_host,sse_host,pa,sse_host,sse_host,sse_host> V1;
  typedef RK4Visitor<ON_HOST,B,S,sse_real,V1,sse_real> Visitor;
  static const int N = net_size<B,S>::value;
  const int P = hostCState.size1();

  #pragma omp parallel
  {
    int id, p;
    real t, h, sgn = (tcur <= tnxt) ? 1.0 : -1.0;
    sse_real t1, h1;
    V1 pax(0);
    sse_real x0[N], x1[N], x2[N], x3[N], x4[N];

    #pragma omp for
    for (p = 0; p < P; p += BI_SSE_SIZE) {
      pax.p = p;
      t = tcur;
      h = sgn*h_h0;

      /* integrate */
      while (sgn*t < sgn*tnxt) {
        /* initialise */
        if (sgn*(t + REAL(1.01)*h - tnxt) > REAL(0.0)) {
          h = tnxt - t;
          if (sgn*h <= REAL(0.0)) {
            t = tnxt;
            break;
          }
        }
        for (id = 0; id < N; ++id) {
          x0[id] = sse_state_get(hostCState, p, id);
        }
        t1 = t;
        h1 = h;

        /* stages */
        Visitor::stage1(t1, h1, pax, x0, x1, x2, x3, x4);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x1[id]);
        }

        Visitor::stage2(t1, h1, pax, x0, x2, x3, x4);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x2[id]);
        }

        Visitor::stage3(t1, h1, pax, x0, x3, x4);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x3[id]);
        }

        Visitor::stage4(t1, h1, pax, x0, x4);
        for (id = 0; id < N; ++id) {
          sse_state_set(hostCState, p, id, x4[id]);
        }

        t += h;
      }
    }
  }
  #else
  typedef typename B::CTypeList S;
  typedef typename boost::mpl::if_c<SH == STATIC_SHARED,const_host,host>::type pa;
  typedef Pa<ON_HOST,B,real,pa,host,host,pa,host,host,host> V1;
  typedef RK4Visitor<ON_HOST,B,S,real,V1,real> Visitor;
  static const int N = net_size<B,S>::value;
  const int P = hostCState.size1();

  #pragma omp parallel
  {
    const real sgn = (tcur <= tnxt) ? 1.0 : -1.0;
    int id, p;
    real t, h;
    V1 pax(0);
    real x0[N], x1[N], x2[N], x3[N], x4[N];

    #pragma omp for
    for (p = 0; p < P; ++p) {
      pax.p = p;
      t = tcur;
      h = sgn*h_h0;

      /* integrate */
      while (sgn*t < sgn*tnxt) {
        /* initialise */
        if (sgn*(t + REAL(1.01)*h - tnxt) > REAL(0.0)) {
          h = tnxt - t;
          if (sgn*h <= REAL(0.0)) {
            t = tnxt;
            break;
          }
        }
        for (id = 0; id < N; ++id) {
          x0[id] = hostCState(p, id);
        }

        /* stages */
        Visitor::stage1(t, h, pax, x0, x1, x2, x3, x4);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x1[id];
        }

        Visitor::stage2(t, h, pax, x0, x2, x3, x4);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x2[id];
        }

        Visitor::stage3(t, h, pax, x0, x3, x4);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x3[id];
        }

        Visitor::stage4(t, h, pax, x0, x4);
        for (id = 0; id < N; ++id) {
          hostCState(pax.p, id) = x4[id];
        }

        t += h;
      }
    }
  }
  #endif
}

#endif

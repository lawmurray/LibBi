/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_RK4INTEGRATOR_HPP
#define BI_ODE_RK4INTEGRATOR_HPP

#include "../misc/location.hpp"
#include "../state/State.hpp"

namespace bi {
/**
 * Update using classic fourth order Runge-Kutta.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class RK4Integrator {
public:
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);

  #ifdef __CUDACC__
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s);
  #endif
};

}

#include "../host/ode/RK4IntegratorHost.hpp"
#ifdef ENABLE_SSE
#include "../sse/ode/RK4IntegratorSSE.hpp"
#endif
#ifdef __CUDACC__
#include "../cuda/ode/RK4IntegratorGPU.cuh"
#endif

template<class B, class S>
template<class T1>
void bi::RK4Integrator<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  if (bi::abs(t2 - t1) > 0.0) {
    #ifdef ENABLE_SSE
    if (s.size() % BI_SIMD_SIZE == 0) {
      RK4IntegratorSSE<B,S,T1>::update(t1, t2, s);
    } else {
      RK4IntegratorHost<B,S,T1>::update(t1, t2, s);
    }
    #else
    RK4IntegratorHost<B,S,T1>::update(t1, t2, s);
    #endif
  }
}

#ifdef __CUDACC__
template<class B, class S>
template<class T1>
void bi::RK4Integrator<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  if (bi::abs(t2 - t1) > 0.0) {
    RK4IntegratorGPU<B,S,T1>::update(t1, t2, s);
  }
}
#endif

#endif

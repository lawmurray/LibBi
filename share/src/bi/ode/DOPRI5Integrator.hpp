/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_DOPRI5INTEGRATOR_HPP
#define BI_ODE_DOPRI5INTEGRATOR_HPP

#include "../misc/location.hpp"
#include "../state/State.hpp"

namespace bi {
/**
 * Update using Dormand-Prince 5(4) integrator with adaptive step-size
 * control.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class DOPRI5Integrator {
public:
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);

  #ifdef __CUDACC__
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s);
  #endif
};

}

#include "../host/ode/DOPRI5IntegratorHost.hpp"
#ifdef ENABLE_SSE
#include "../sse/ode/DOPRI5IntegratorSSE.hpp"
#endif
#ifdef __CUDACC__
#include "../cuda/ode/DOPRI5IntegratorGPU.cuh"
#endif

template<class B, class S>
template<class T1>
void bi::DOPRI5Integrator<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  if (bi::abs(t2 - t1) > 0.0) {
    #ifdef ENABLE_SSE
    if (s.size() % BI_SIMD_SIZE == 0) {
      DOPRI5IntegratorSSE<B,S,T1>::update(t1, t2, s);
    } else {
      DOPRI5IntegratorHost<B,S,T1>::update(t1, t2, s);
    }
    #else
    DOPRI5IntegratorHost<B,S,T1>::update(t1, t2, s);
    #endif
  }
}

#ifdef __CUDACC__
template<class B, class S>
template<class T1>
void bi::DOPRI5Integrator<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  if (bi::abs(t2 - t1) > 0.0) {
    DOPRI5IntegratorGPU<B,S,T1>::update(t1, t2, s);
  }
}
#endif

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_RK43INTEGRATOR_HPP
#define BI_ODE_RK43INTEGRATOR_HPP

#include "../misc/location.hpp"
#include "../state/State.hpp"

namespace bi {
/**
 * Update using RK4(3)5[2R+]C low-storage integrator with adaptive step-size
 * control.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 */
template<class B, class S>
class RK43Integrator {
public:
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_HOST>& s);

  #ifdef __CUDACC__
  template<class T1>
  static void update(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s);
  #endif
};

}

#include "../host/ode/RK43IntegratorHost.hpp"
#ifdef ENABLE_SSE
#include "../sse/ode/RK43IntegratorSSE.hpp"
#endif
#ifdef __CUDACC__
#include "../cuda/ode/RK43IntegratorGPU.cuh"
#endif

template<class B, class S>
template<class T1>
void bi::RK43Integrator<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_HOST>& s) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  if (bi::abs(t2 - t1) > 0.0) {
    #ifdef ENABLE_SSE
    if (s.size() % BI_SIMD_SIZE == 0) {
      RK43IntegratorSSE<B,S,T1>::update(t1, t2, s);
    } else {
      RK43IntegratorHost<B,S,T1>::update(t1, t2, s);
    }
    #else
    RK43IntegratorHost<B,S,T1>::update(t1, t2, s);
    #endif
  }
}

#ifdef __CUDACC__
template<class B, class S>
template<class T1>
void bi::RK43Integrator<B,S>::update(const T1 t1, const T1 t2,
    State<B,ON_DEVICE>& s) {
  /* pre-conditions */
  BI_ASSERT(t1 <= t2);

  if (bi::abs(t2 - t1) > 0.0) {
    RK43IntegratorGPU<B,S,T1>::update(t1, t2, s);
  }
}
#endif

#endif

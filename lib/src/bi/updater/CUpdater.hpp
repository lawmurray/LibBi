/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_CUPDATER_HPP
#define BI_UPDATER_CUPDATER_HPP

#include "../cuda/cuda.hpp"
#include "../method/misc.hpp"

namespace bi {
/**
 * @internal
 *
 * Updater for c-net.
 *
 * @ingroup method_updater
 *
 * @tparam B Model type.
 * @tparam SH Static handling.
 */
template<class B, StaticHandling SH = STATIC_SHARED>
class CUpdater {
public:
  /**
   * Update c-net.
   *
   * @param t Current time.
   * @param tnxt Time to which to step forward.
   * @param s State to update.
   */
  void update(const real t, const real tnxt, State<ON_HOST>& s);

  /**
   * @copydoc update(const real, const real, State<ON_HOST>&)
   */
  void update(const real t, const real tnxt, State<ON_DEVICE>& s);
};
}

#include "../host/bind.hpp"
#if defined(USE_ODE_DOPRI5)
#include "../ode/DOPRI5Integrator.hpp"
#elif defined(USE_ODE_RK4)
#include "../ode/RK4Integrator.hpp"
#else
#include "../ode/RK43Integrator.hpp"
#endif

template<class B, bi::StaticHandling SH>
void bi::CUpdater<B,SH>::update(const real t, const real tnxt,
    State<ON_HOST>& s) {
  typedef typename B::CTypeList S;

  if (net_size<B,S>::value && t < tnxt) {
    bind(s);
    #if defined(USE_ODE_DOPRI5)
    DOPRI5Integrator<ON_HOST,B,(unsigned)SH>::stepTo(t, tnxt);
    #elif defined(USE_ODE_RK4)
    RK4Integrator<ON_HOST,B,(unsigned)SH>::stepTo(t, tnxt);
    #else
    RK43Integrator<ON_HOST,B,(unsigned)SH>::stepTo(t, tnxt);
    #endif
    unbind(s);
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/CUpdater.cuh"
#endif

#endif

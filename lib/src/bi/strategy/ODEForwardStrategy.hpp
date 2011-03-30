/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_ODEFORWARDSTRATEGY_HPP
#define BI_STRATEGY_ODEFORWARDSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for forward update with IS_ODE_FORWARD.
 *
 * @ingroup method_strategy
 *
 * @copydoc ForwardStrategy
 */
template<class X, class T, class V1, class V2>
class ODEForwardStrategy {
public:
  /**
   * @copydoc ForwardStrategy::f()
   *
   * Performs a single Euler step over the time interval.
   *
   * @note Simulator and CUpdater do cleverer updates of IS_ODE_FORWARD nodes
   * using Runge-Kutta. This simpler strategy is generally not used, but
   * provided for completeness.
   */
  static CUDA_FUNC_BOTH void f(const Coord& cox, const T t, const V1& pax,
      const T& tnxt, V2& xnxt) {
    real dx;
    X::dfdt(cox, t, pax, dx);
    xnxt = pax.template fetch<X>(cox) + (tnxt - t)*dx;
  }

};

}

#endif

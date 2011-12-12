/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_NOFORWARDSTRATEGY_HPP
#define BI_STRATEGY_NOFORWARDSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Default forward strategy (do nothing).
 *
 * @ingroup method_strategy
 *
 * @copydoc ForwardStrategy
 */
template<class X, class T, class V1, class V2>
class NoForwardStrategy {
public:
  /**
   * @copydoc ForwardStrategy::f()
   */
  static CUDA_FUNC_BOTH void f(const Coord& cox, const T t, const V1& pax,
      const T tnxt, V2& xnxt) {
    //
  }

};

}

#endif

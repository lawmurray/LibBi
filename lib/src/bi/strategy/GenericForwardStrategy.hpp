/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_GENERICFORWARDSTRATEGY_HPP
#define BI_STRATEGY_GENERICFORWARDSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for generic forward updates.
 *
 * @ingroup method_strategy
 *
 * @copydoc ForwardStrategy
 */
template<class X, class T, class V1, class V2>
class GenericForwardStrategy {
public:
  /**
   * @copydoc ForwardStrategy::f()
   */
  static CUDA_FUNC_BOTH void f(const Coord& cox, const T t, const V1& pax,
      const T tnxt, V2& xnxt);

};

}

template<class X, class T, class V1, class V2>
inline void bi::GenericForwardStrategy<X,T,V1,V2>::f(const Coord& cox,
    const T t, const V1& pax, const T tnxt, V2& xnxt) {
  X::f(cox, t, pax, tnxt, xnxt);
}

#endif

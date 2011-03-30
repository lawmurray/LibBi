/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_GENERICSTATICSTRATEGY_HPP
#define BI_STRATEGY_GENERICSTATICSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for static samples.
 *
 * @ingroup method_strategy
 *
 * @copydoc StaticStrategy
 */
template<class X, class V1, class V2>
class GenericStaticStrategy {
public:
  /**
   * @copydoc StaticStrategy::s()
   */
  static CUDA_FUNC_BOTH void s(const Coord& cox, const V1& pax, V2& x) {
    X::s(cox, pax, x);
  }

};

}

#endif

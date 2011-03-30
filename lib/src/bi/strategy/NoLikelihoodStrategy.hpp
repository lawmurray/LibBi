/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_NOLIKELIHOODSTRATEGY_HPP
#define BI_STRATEGY_NOLIKELIHOODSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * @ingroup method_strategy
 *
 * @copydoc LikelihoodStrategy
 */
template<class X, class V1, class V2, class V3>
class NoLikelihoodStrategy {
public:
  /**
   * @copydoc LikelihoodStrategy::l()
   */
  static CUDA_FUNC_BOTH void l(const Coord& cox, const V1& pax, const V2& y,
      V3& l) {
    l = static_cast<V3>(1.0);
  }

};

}

#endif

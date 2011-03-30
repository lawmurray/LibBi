/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_NOOBSERVATIONSTRATEGY_HPP
#define BI_STRATEGY_NOOBSERVATIONSTRATEGY_HPP

#include "../cuda/cuda.hpp"
#include "../traits/likelihood_traits.hpp"
#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * @ingroup method_strategy
 *
 * @copydoc ObservationStrategy
 */
template<class X, class V1, class V2>
class NoObservationStrategy {
public:
  /**
   * @copydoc ObservationStrategy::o()
   */
  static CUDA_FUNC_BOTH void o(const real r, const Coord& cox,
      const V1& pax, V2& x) {
    //
  }

};
}

#endif

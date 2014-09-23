/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_HANDLERFACTORY_HPP
#define BI_METHOD_HANDLERFACTORY_HPP

#include "MarginalSISHandler.hpp"

namespace bi {
/**
 * Handler factory.
 *
 * @ingroup server
 */
class HandlerFactory {
public:
  /**
   * Create handler for marginal sequential importance sampling.
   */
  template<class B, class A, class S>
  static MarginalSISHandler<B,A,S>* createMarginalSISHandler(B& m,
      const int T, A& adapter, S& stopper, TreeNetworkNode& node);
};
}

template<class B, class A, class S>
bi::MarginalSISHandler<B,A,S>* bi::HandlerFactory::createMarginalSISHandler(
    B& m, const int T, A& adapter, S& stopper, TreeNetworkNode& node) {
  return new MarginalSISHandler<B,A,S>(m, T, adapter, stopper, node);
}

#endif

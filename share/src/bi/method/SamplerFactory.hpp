/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SAMPLERFACTORY_HPP
#define BI_METHOD_SAMPLERFACTORY_HPP

#include "MarginalMH.hpp"

namespace bi {
/**
 * Sampler factory.
 *
 * @ingroup method_sampler
 */
class SamplerFactory {
public:
  /**
   * Create marginal Metropolis--Hastings sampler.
   */
  template<class B, class F>
  static MarginalMH<B,F>* createMarginalMH(B& m, F& filter);
};
}

template<class B, class F>
bi::MarginalMH<B,F>* bi::SamplerFactory::createMarginalMH(B& m, F& filter) {
  return new MarginalMH<B,F>(m, filter);
}

#endif

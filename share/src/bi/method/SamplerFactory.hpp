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
#include "MarginalSIR.hpp"
#include "MarginalSRS.hpp"

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

  /**
   * Create marginal sequential importance resampling sampler.
   */
  template<class B, class F, class A, class R>
  static MarginalSIR<B,F,A,R>* createMarginalSIR(B& m, F& mmh, A& adapter,
      R& resam, const int Nmoves = 1);

  /**
   * Create marginal sequential rejection sampler.
   */
  template<class B, class F, class A, class S>
  static MarginalSRS<B,F,A,S>* createMarginalSRS(B& m, F& filter, A& adapter,
      S& stopper);
};
}

template<class B, class F>
bi::MarginalMH<B,F>* bi::SamplerFactory::createMarginalMH(B& m, F& filter) {
  return new MarginalMH<B,F>(m, filter);
}

template<class B, class F, class A, class R>
bi::MarginalSIR<B,F,A,R>* bi::SamplerFactory::createMarginalSIR(B& m, F& mmh,
    A& adapter, R& resam, const int Nmoves) {
  return new MarginalSIR<B,F,A,R>(m, mmh, adapter, resam, Nmoves);
}

template<class B, class F, class A, class S>
bi::MarginalSRS<B,F,A,S>* bi::SamplerFactory::createMarginalSRS(B& m,
    F& filter, A& adapter, S& stopper) {
  return new MarginalSRS<B,F,A,S>(m, filter, adapter, stopper);
}

#endif

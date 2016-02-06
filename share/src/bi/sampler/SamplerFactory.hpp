/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SAMPLER_SAMPLERFACTORY_HPP
#define BI_SAMPLER_SAMPLERFACTORY_HPP

#include "MarginalMH.hpp"
#include "MarginalSIR.hpp"
#include "MarginalSIS.hpp"

#include "boost/shared_ptr.hpp"

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
  static boost::shared_ptr<MarginalMH<B,F> > createMarginalMH(B& m,
      F& filter);

  /**
   * Create marginal sequential importance resampling sampler.
   */
  template<class B, class F, class A, class R>
  static boost::shared_ptr<MarginalSIR<B,F,A,R> > createMarginalSIR(B& m,
      F& mmh, A& adapter, R& resam, const int nmoves = 1,
      const double tmoves = 0.0);

  /**
   * Create marginal sequential rejection sampler.
   */
  template<class B, class F, class A, class S>
  static boost::shared_ptr<MarginalSIS<B,F,A,S> > createMarginalSIS(B& m,
      F& filter, A& adapter, S& stopper);
};
}

template<class B, class F>
boost::shared_ptr<bi::MarginalMH<B,F> > bi::SamplerFactory::createMarginalMH(
    B& m, F& filter) {
  return boost::shared_ptr < MarginalMH<B,F>
      > (new MarginalMH<B,F>(m, filter));
}

template<class B, class F, class A, class R>
boost::shared_ptr<bi::MarginalSIR<B,F,A,R> > bi::SamplerFactory::createMarginalSIR(
    B& m, F& mmh, A& adapter, R& resam, const int nmoves,
    const double tmoves) {
  return boost::shared_ptr < MarginalSIR<B,F,A,R>
      > (new MarginalSIR<B,F,A,R>(m, mmh, adapter, resam, nmoves, tmoves));
}

template<class B, class F, class A, class S>
boost::shared_ptr<bi::MarginalSIS<B,F,A,S> > bi::SamplerFactory::createMarginalSIS(
    B& m, F& filter, A& adapter, S& stopper) {
  return boost::shared_ptr < MarginalSIS<B,F,A,S>
      > (new MarginalSIS<B,F,A,S>(m, filter, adapter, stopper));
}

#endif

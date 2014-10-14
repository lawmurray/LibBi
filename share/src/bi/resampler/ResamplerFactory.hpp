/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_RESAMPLERFACTORY_HPP
#define BI_RESAMPLER_RESAMPLERFACTORY_HPP

#include "MultinomialResampler.hpp"
#include "StratifiedResampler.hpp"
#include "SystematicResampler.hpp"
#include "MetropolisResampler.hpp"
#include "RejectionResampler.hpp"
#ifdef ENABLE_MPI
#include "../mpi/resampler/DistributedResampler.hpp"
#endif

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Resampler factory.
 *
 * @ingroup method_resampler
 */
class ResamplerFactory {
public:
  /**
   * Create multinomial resampler.
   */
  static boost::shared_ptr<MultinomialResampler> createMultinomialResampler(
      const double essRel = 0.5);

  /**
   * Create stratified resampler.
   */
  static boost::shared_ptr<StratifiedResampler> createStratifiedResampler(
      const double essRel = 0.5);

  /**
   * Create systematic resampler.
   */
  static boost::shared_ptr<SystematicResampler> createSystematicResampler(
      const double essRel = 0.5);

  /**
   * Create Metropolis resampler.
   */
  static boost::shared_ptr<MetropolisResampler> createMetropolisResampler(
      const int B, const double essRel = 0.5);

  /**
   * Create rejection resampler.
   */
  static boost::shared_ptr<RejectionResampler> createRejectionResampler();

  #ifdef ENABLE_MPI
  /**
   * Create distributed resampler.
   */
  template<class R>
  static boost::shared_ptr<DistributedResampler<R> > createDistributedResampler(
      boost::smart_ptr<R> base, const double essRel = 0.5);
  #endif
};
}

#ifdef ENABLE_MPI
template<class R>
boost::shared_ptr<bi::DistributedResampler<R> > bi::ResamplerFactory::createDistributedResampler(
    boost::smart_ptr<R> base, const double essRel) {
  return new DistributedResampler<R>(base, essRel);
}
#endif

#endif

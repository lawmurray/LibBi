/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_RESAMPLER_DISTRIBUTEDRESAMPLERFACTORY_HPP
#define BI_RESAMPLER_DISTRIBUTEDRESAMPLERFACTORY_HPP

#include "DistributedResampler.hpp"
#include "../../resampler/MultinomialResampler.hpp"
#include "../../resampler/StratifiedResampler.hpp"
#include "../../resampler/SystematicResampler.hpp"
#include "../../resampler/MetropolisResampler.hpp"
#include "../../resampler/RejectionResampler.hpp"

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"

namespace bi {
/**
 * Resampler factory.
 *
 * @ingroup method_resampler
 */
class DistributedResamplerFactory {
public:
  /**
   * Create multinomial resampler.
   */
  static boost::shared_ptr<DistributedResampler<MultinomialResampler> > createMultinomialResampler(
      const double essRel = 0.5, const bool anytime = false);

  /**
   * Create stratified resampler.
   */
  static boost::shared_ptr<DistributedResampler<StratifiedResampler> > createStratifiedResampler(
      const double essRel = 0.5, const bool anytime = false);

  /**
   * Create systematic resampler.
   */
  static boost::shared_ptr<DistributedResampler<SystematicResampler> > createSystematicResampler(
      const double essRel = 0.5, const bool anytime = false);

  /**
   * Create Metropolis resampler.
   */
  static boost::shared_ptr<DistributedResampler<MetropolisResampler> > createMetropolisResampler(
      const int B, const double essRel = 0.5, const bool anytime = false);

  /**
   * Create rejection resampler.
   */
  static boost::shared_ptr<DistributedResampler<RejectionResampler> > createRejectionResampler(
      const bool anytime = false);
};
}

#endif

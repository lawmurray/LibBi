/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_RESAMPLER_RESAMPLERFACTORY_HPP
#define BI_RESAMPLER_RESAMPLERFACTORY_HPP

#include "Resampler.hpp"
#include "MultinomialResampler.hpp"
#include "StratifiedResampler.hpp"
#include "SystematicResampler.hpp"
#include "MetropolisResampler.hpp"
#include "RejectionResampler.hpp"

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"

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
  static boost::shared_ptr<Resampler<MultinomialResampler> > createMultinomialResampler(
      const double essRel = 0.5, const bool anytime = false);

  /**
   * Create stratified resampler.
   */
  static boost::shared_ptr<Resampler<StratifiedResampler> > createStratifiedResampler(
      const double essRel = 0.5, const bool anytime = false);

  /**
   * Create systematic resampler.
   */
  static boost::shared_ptr<Resampler<SystematicResampler> > createSystematicResampler(
      const double essRel = 0.5, const bool anytime = false);

  /**
   * Create Metropolis resampler.
   */
  static boost::shared_ptr<Resampler<MetropolisResampler> > createMetropolisResampler(
      const int B, const double essRel = 0.5, const bool anytime = false);

  /**
   * Create rejection resampler.
   */
  static boost::shared_ptr<Resampler<RejectionResampler> > createRejectionResampler(
      const bool anytime = false);
};
}

#endif

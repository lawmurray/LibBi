/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#include "ResamplerFactory.hpp"

boost::shared_ptr<bi::Resampler<bi::MultinomialResampler> > bi::ResamplerFactory::createMultinomialResampler(
    const double essRel, const bool anytime) {
  return boost::make_shared < Resampler<MultinomialResampler>
      > (essRel, anytime);
}

boost::shared_ptr<bi::Resampler<bi::StratifiedResampler> > bi::ResamplerFactory::createStratifiedResampler(
    const double essRel, const bool anytime) {
  return boost::make_shared < Resampler<StratifiedResampler>
      > (essRel, anytime);
}

boost::shared_ptr<bi::Resampler<bi::SystematicResampler> > bi::ResamplerFactory::createSystematicResampler(
    const double essRel, const bool anytime) {
  return boost::make_shared < Resampler<SystematicResampler>
      > (essRel, anytime);
}

boost::shared_ptr<bi::Resampler<bi::MetropolisResampler> > bi::ResamplerFactory::createMetropolisResampler(
    const int B, const double essRel, const bool anytime) {
  BOOST_AUTO(resam,
      boost::make_shared < Resampler<MetropolisResampler>
          > (essRel, anytime));
  resam->setSteps(B);
  return resam;
}

boost::shared_ptr<bi::Resampler<bi::RejectionResampler> > bi::ResamplerFactory::createRejectionResampler(
    const bool anytime) {
  return boost::make_shared < Resampler<RejectionResampler> > (1.0, anytime);
}

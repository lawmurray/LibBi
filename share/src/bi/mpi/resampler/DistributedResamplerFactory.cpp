/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#include "DistributedResamplerFactory.hpp"

boost::shared_ptr<bi::DistributedResampler<bi::MultinomialResampler> > bi::DistributedResamplerFactory::createMultinomialResampler(
    const double essRel, const bool anytime) {
  return boost::make_shared < DistributedResampler<MultinomialResampler>
      > (essRel, anytime);
}

boost::shared_ptr<bi::DistributedResampler<bi::StratifiedResampler> > bi::DistributedResamplerFactory::createStratifiedResampler(
    const double essRel, const bool anytime) {
  return boost::make_shared < DistributedResampler<StratifiedResampler>
      > (essRel, anytime);
}

boost::shared_ptr<bi::DistributedResampler<bi::SystematicResampler> > bi::DistributedResamplerFactory::createSystematicResampler(
    const double essRel, const bool anytime) {
  return boost::make_shared < DistributedResampler<SystematicResampler>
      > (essRel, anytime);
}

boost::shared_ptr<bi::DistributedResampler<bi::MetropolisResampler> > bi::DistributedResamplerFactory::createMetropolisResampler(
    const int B, const double essRel, const bool anytime) {
  BOOST_AUTO(resam,
      boost::make_shared < DistributedResampler<MetropolisResampler>
          > (essRel, anytime));
  resam->setSteps(B);
  return resam;
}

boost::shared_ptr<bi::DistributedResampler<bi::RejectionResampler> > bi::DistributedResamplerFactory::createRejectionResampler(
    const bool anytime) {
  return boost::make_shared < DistributedResampler<RejectionResampler>
      > (1.0, anytime);
}

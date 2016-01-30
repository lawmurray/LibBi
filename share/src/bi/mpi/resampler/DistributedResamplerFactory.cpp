/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#include "DistributedResamplerFactory.hpp"

boost::shared_ptr<bi::DistributedResampler<bi::MultinomialResampler> > bi::DistributedResamplerFactory::createMultinomialResampler(
    const double essRel) {
  return boost::make_shared < DistributedResampler<MultinomialResampler>
      > (essRel);
}

boost::shared_ptr<bi::DistributedResampler<bi::StratifiedResampler> > bi::DistributedResamplerFactory::createStratifiedResampler(
    const double essRel) {
  return boost::make_shared < DistributedResampler<StratifiedResampler>
      > (essRel);
}

boost::shared_ptr<bi::DistributedResampler<bi::SystematicResampler> > bi::DistributedResamplerFactory::createSystematicResampler(
    const double essRel) {
  return boost::make_shared < DistributedResampler<SystematicResampler>
      > (essRel);
}

boost::shared_ptr<bi::DistributedResampler<bi::MetropolisResampler> > bi::DistributedResamplerFactory::createMetropolisResampler(
    const int B, const double essRel) {
  BOOST_AUTO(resam,
      boost::make_shared < DistributedResampler<MetropolisResampler>
          > (essRel));
  resam->setSteps(B);
  return resam;
}

boost::shared_ptr<bi::DistributedResampler<bi::RejectionResampler> > bi::DistributedResamplerFactory::createRejectionResampler() {
  return boost::make_shared < DistributedResampler<RejectionResampler> > (1.0);
}

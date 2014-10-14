/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ResamplerFactory.hpp"

boost::shared_ptr<bi::MultinomialResampler> bi::ResamplerFactory::createMultinomialResampler(
    const double essRel) {
  return boost::shared_ptr<MultinomialResampler>(new MultinomialResampler(essRel));
}

boost::shared_ptr<bi::StratifiedResampler> bi::ResamplerFactory::createStratifiedResampler(
    const double essRel) {
  return boost::shared_ptr<StratifiedResampler>(new StratifiedResampler(essRel));
}

boost::shared_ptr<bi::SystematicResampler> bi::ResamplerFactory::createSystematicResampler(
    const double essRel) {
  return boost::shared_ptr<SystematicResampler>(new SystematicResampler(essRel));
}

boost::shared_ptr<bi::MetropolisResampler> bi::ResamplerFactory::createMetropolisResampler(
    const int B, const double essRel) {
  return boost::shared_ptr<MetropolisResampler>(new MetropolisResampler(B, essRel));
}

boost::shared_ptr<bi::RejectionResampler> bi::ResamplerFactory::createRejectionResampler() {
  return boost::shared_ptr<RejectionResampler>(new RejectionResampler());
}

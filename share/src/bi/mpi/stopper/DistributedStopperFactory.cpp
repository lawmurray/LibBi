/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#include "DistributedStopperFactory.hpp"

#include "boost/make_shared.hpp"

boost::shared_ptr<bi::DistributedStopper<bi::DefaultStopper> > bi::DistributedStopperFactory::createDefaultStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < DistributedStopper<DefaultStopper>
      > (threshold, maxP, T);
}

boost::shared_ptr<bi::DistributedStopper<bi::MinimumESSStopper> > bi::DistributedStopperFactory::createMinimumESSStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < DistributedStopper<MinimumESSStopper>
      > (threshold, maxP, T);
}

boost::shared_ptr<bi::DistributedStopper<bi::StdDevStopper> > bi::DistributedStopperFactory::createStdDevStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < DistributedStopper<StdDevStopper>
      > (threshold, maxP, T);
}

boost::shared_ptr<bi::DistributedStopper<bi::SumOfWeightsStopper> > bi::DistributedStopperFactory::createSumOfWeightsStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < DistributedStopper<SumOfWeightsStopper>
      > (threshold, maxP, T);
}

boost::shared_ptr<bi::DistributedStopper<bi::VarStopper> > bi::DistributedStopperFactory::createVarStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < DistributedStopper<VarStopper>
      > (threshold, maxP, T);
}

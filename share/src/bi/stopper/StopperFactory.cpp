/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "StopperFactory.hpp"

#include "boost/make_shared.hpp"

boost::shared_ptr<bi::Stopper<bi::DefaultStopper> > bi::StopperFactory::createDefaultStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < Stopper<DefaultStopper> > (threshold, maxP, T);
}

boost::shared_ptr<bi::Stopper<bi::MinimumESSStopper> > bi::StopperFactory::createMinimumESSStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < Stopper<MinimumESSStopper>
      > (threshold, maxP, T);
}

boost::shared_ptr<bi::Stopper<bi::StdDevStopper> > bi::StopperFactory::createStdDevStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < Stopper<StdDevStopper> > (threshold, maxP, T);
}

boost::shared_ptr<bi::Stopper<bi::SumOfWeightsStopper> > bi::StopperFactory::createSumOfWeightsStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < Stopper<SumOfWeightsStopper>
      > (threshold, maxP, T);
}

boost::shared_ptr<bi::Stopper<bi::VarStopper> > bi::StopperFactory::createVarStopper(
    const double threshold, const int maxP, const int T) {
  return boost::make_shared < Stopper<VarStopper> > (threshold, maxP, T);
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "StopperFactory.hpp"

boost::shared_ptr<bi::MinimumESSStopper> bi::StopperFactory::createMinimumESSStopper(
    const double threshold, const int maxP, const int T) {
  return boost::shared_ptr < MinimumESSStopper
      > (new MinimumESSStopper(threshold, maxP, T));
}

boost::shared_ptr<bi::StdDevStopper> bi::StopperFactory::createStdDevStopper(
    const double threshold, const int maxP, const int T) {
  return boost::shared_ptr < StdDevStopper
      > (new StdDevStopper(threshold, maxP, T));
}

boost::shared_ptr<bi::Stopper> bi::StopperFactory::createStopper(
    const double threshold, const int maxP, const int T) {
  return boost::shared_ptr < Stopper > (new Stopper(threshold, maxP, T));
}

boost::shared_ptr<bi::SumOfWeightsStopper> bi::StopperFactory::createSumOfWeightsStopper(
    const double threshold, const int maxP, const int T) {
  return boost::shared_ptr < SumOfWeightsStopper
      > (new SumOfWeightsStopper(threshold, maxP, T));
}

boost::shared_ptr<bi::VarStopper> bi::StopperFactory::createVarStopper(
    const double threshold, const int maxP, const int T) {
  return boost::shared_ptr < VarStopper > (new VarStopper(threshold, maxP, T));
}

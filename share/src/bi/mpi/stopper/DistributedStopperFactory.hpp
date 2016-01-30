/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPERFACTORY_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPERFACTORY_HPP

#include "DistributedStopper.hpp"
#include "../../stopper/DefaultStopper.hpp"
#include "../../stopper/MinimumESSStopper.hpp"
#include "../../stopper/StdDevStopper.hpp"
#include "../../stopper/SumOfWeightsStopper.hpp"
#include "../../stopper/VarStopper.hpp"

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Distributed stopper factory.
 *
 * @ingroup method_stopper
 */
class DistributedStopperFactory {
public:
  static boost::shared_ptr<DistributedStopper<DefaultStopper> > createDefaultStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<DistributedStopper<MinimumESSStopper> > createMinimumESSStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<DistributedStopper<StdDevStopper> > createStdDevStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<DistributedStopper<SumOfWeightsStopper> > createSumOfWeightsStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<DistributedStopper<VarStopper> > createVarStopper(
      const double threshold, const int maxP, const int T);
};
}

#endif

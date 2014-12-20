/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STOPPER_STOPPERFACTORY_HPP
#define BI_STOPPER_STOPPERFACTORY_HPP

#include "MinimumESSStopper.hpp"
#include "StdDevStopper.hpp"
#include "Stopper.hpp"
#include "SumOfWeightsStopper.hpp"
#include "VarStopper.hpp"
#ifdef ENABLE_MPI
#include "../mpi/stopper/DistributedStopper.hpp"
#endif

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Stopper factory.
 *
 * @ingroup method_stopper
 */
class StopperFactory {
public:
  static boost::shared_ptr<MinimumESSStopper> createMinimumESSStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<StdDevStopper> createStdDevStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<Stopper> createStopper(const double threshold,
      const int maxP, const int T);

  static boost::shared_ptr<SumOfWeightsStopper> createSumOfWeightsStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<VarStopper> createVarStopper(
      const double threshold, const int maxP, const int T);

#ifdef ENABLE_MPI
  /**
   * Create distributed stopper.
   */
  template<class S>
  static boost::shared_ptr<DistributedStopper<S> > createDistributedStopper(
      boost::smart_ptr<S> base, TreeNetworkNode& node);
#endif
};
}

#ifdef ENABLE_MPI
template<class S>
boost::shared_ptr<bi::DistributedStopper<S> > bi::StopperFactory::createDistributedStopper(
    boost::smart_ptr<R> base, TreeNetworkNode& node) {
  return new DistributedStopper<S>(base, node);
}
#endif

#endif

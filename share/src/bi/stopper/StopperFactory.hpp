/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STOPPER_STOPPERFACTORY_HPP
#define BI_STOPPER_STOPPERFACTORY_HPP

#include "Stopper.hpp"
#include "DefaultStopper.hpp"
#include "MinimumESSStopper.hpp"
#include "StdDevStopper.hpp"
#include "SumOfWeightsStopper.hpp"
#include "VarStopper.hpp"

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Stopper factory.
 *
 * @ingroup method_stopper
 */
class StopperFactory {
public:
  static boost::shared_ptr<Stopper<DefaultStopper> > createDefaultStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<Stopper<MinimumESSStopper> > createMinimumESSStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<Stopper<StdDevStopper> > createStdDevStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<Stopper<SumOfWeightsStopper> > createSumOfWeightsStopper(
      const double threshold, const int maxP, const int T);

  static boost::shared_ptr<Stopper<VarStopper> > createVarStopper(
      const double threshold, const int maxP, const int T);
};
}

#endif

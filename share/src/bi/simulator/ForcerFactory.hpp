/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SIMULATOR_FORCERFACTORY_HPP
#define BI_SIMULATOR_FORCERFACTORY_HPP

#include "Forcer.hpp"

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Factory for creating Forcer objects.
 *
 * @ingroup method_simulator
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct ForcerFactory {
  /**
   * Create Forcer.
   *
   * @return Forcer object. Caller has ownership.
   *
   * @see Forcer::Forcer()
   */
  template<class IO1>
  static boost::shared_ptr<Forcer<IO1,CL> > create(IO1& in) {
    return boost::shared_ptr<Forcer<IO1,CL> >(new Forcer<IO1,CL>(in));
  }
};
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SIMULATOR_OBSERVERFACTORY_HPP
#define BI_SIMULATOR_OBSERVERFACTORY_HPP

#include "Observer.hpp"

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Factory for creating Observer objects.
 *
 * @ingroup method
 *
 * @see Observer
 */
template<Location CL = ON_HOST>
struct ObserverFactory {
  /**
   * Create observer.
   *
   * @return Observer object. Caller has ownership.
   *
   * @see Observer::Observer()
   */
  template<class IO1>
  static boost::shared_ptr<Observer<IO1,CL> > create(IO1& in) {
    return boost::shared_ptr<Observer<IO1,CL> >(new Observer<IO1,CL>(in));
  }
};
}

#endif

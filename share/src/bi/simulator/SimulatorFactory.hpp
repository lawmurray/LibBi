/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_SIMULATORFACTORY_HPP
#define BI_METHOD_SIMULATORFACTORY_HPP

#include "Simulator.hpp"

#include "boost/shared_ptr.hpp"

namespace bi {
/**
 * Simulator factory.
 *
 * @ingroup method
 */
class SimulatorFactory {
public:
  /**
   * Create simulator.
   */
  template<class B, class F, class O>
  static boost::shared_ptr<Simulator<B,F,O> > create(B& m, F& in, O& obs);
};
}

template<class B, class F, class O>
boost::shared_ptr<bi::Simulator<B,F,O> > bi::SimulatorFactory::create(B& m,
    F& in, O& obs) {
  return boost::shared_ptr<Simulator<B,F,O> >(new Simulator<B,F,O>(m, in, obs));
}

#endif

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
  static Simulator<B,F,O>* create(B& m, F& in, O& obs);
};
}

template<class B, class F, class O>
bi::Simulator<B,F,O>* bi::SimulatorFactory::create(B& m, F& in, O& obs) {
  return new Simulator<B,F,O>(m, in, obs);
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_HYPER_HPP
#define BI_PROGRAM_HYPER_HPP

#include "Var.hpp"

namespace biprog {
/**
 * Hyperparameter.
 *
 * @ingroup program
 */
class Hyper: public Var {
public:
  /**
   * Constructor.
   */
  Hyper();

  /**
   * Destructor.
   */
  virtual ~Hyper();
};
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_OBS_HPP
#define BI_PROGRAM_OBS_HPP

#include "Var.hpp"

namespace biprog {
/**
 * Observed variable.
 *
 * @ingroup program
 */
class Obs: public Var {
public:
  /**
   * Constructor.
   */
  Obs();

  /**
   * Destructor.
   */
  virtual ~Obs();
};
}

#endif

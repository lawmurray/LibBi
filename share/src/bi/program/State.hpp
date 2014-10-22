/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_STATE_HPP
#define BI_PROGRAM_STATE_HPP

#include "Var.hpp"

namespace biprog {
/**
 * State variable.
 *
 * @ingroup program
 */
class State: public Var {
public:
  /**
   * Constructor.
   */
  State();

  /**
   * Destructor.
   */
  virtual ~State();
};
}

#endif

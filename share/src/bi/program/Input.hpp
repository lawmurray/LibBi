/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_INPUT_HPP
#define BI_PROGRAM_INPUT_HPP

#include "Var.hpp"

namespace biprog {
/**
 * Input variable.
 *
 * @ingroup program
 */
class Input: public Var {
public:
  /**
   * Constructor.
   */
  Input();

  /**
   * Destructor.
   */
  virtual ~Input();
};
}

#endif

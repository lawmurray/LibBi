/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PARAM_HPP
#define BI_PROGRAM_PARAM_HPP

#include "Var.hpp"

namespace biprog {
/**
 * Parameter.
 *
 * @ingroup program
 */
class Param: public Var {
public:
  /**
   * Constructor.
   */
  Param();

  /**
   * Destructor.
   */
  virtual ~Param();
};
}

#endif

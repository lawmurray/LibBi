/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_VAR_HPP
#define BI_PROGRAM_VAR_HPP

#include "Declared.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public Declared {
public:
  /**
   * Constructor.
   */
  Var();

  /**
   * Destructor.
   */
  virtual ~Var();
};
}

#endif

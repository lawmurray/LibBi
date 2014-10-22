/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_FUNCTION_HPP
#define BI_PROGRAM_FUNCTION_HPP

#include "Declared.hpp"

namespace biprog {
/**
 * Function.
 *
 * @ingroup program
 */
class Function: public Declared {
public:
  /**
   * Constructor.
   */
  Function();

  /**
   * Destructor.
   */
  virtual ~Function();
};
}

#endif

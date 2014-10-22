/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONAL_HPP
#define BI_PROGRAM_CONDITIONAL_HPP

#include "Declared.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public Declared {
public:
  /**
   * Constructor.
   */
  Conditional();

  /**
   * Destructor.
   */
  virtual ~Conditional();
};
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_INTLITERAL_HPP
#define BI_PROGRAM_INTLITERAL_HPP

#include "Literal.hpp"

namespace biprog {
/**
 * Int literal.
 *
 * @ingroup program
 */
class IntLiteral: public Literal {
public:
  /**
   * Constructor.
   */
  IntLiteral();

  /**
   * Destructor.
   */
  virtual ~IntLiteral();
};
}

#endif

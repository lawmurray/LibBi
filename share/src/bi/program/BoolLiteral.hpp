/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BOOLLITERAL_HPP
#define BI_PROGRAM_BOOLLITERAL_HPP

#include "Literal.hpp"

namespace biprog {
/**
 * Bool literal.
 *
 * @ingroup program
 */
class BoolLiteral: public Literal {
public:
  /**
   * Constructor.
   */
  BoolLiteral();

  /**
   * Destructor.
   */
  virtual ~BoolLiteral();
};
}

#endif

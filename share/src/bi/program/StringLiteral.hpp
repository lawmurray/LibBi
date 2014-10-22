/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_STRINGLITERAL_HPP
#define BI_PROGRAM_STRINGLITERAL_HPP

#include "Literal.hpp"

namespace biprog {
/**
 * String literal.
 *
 * @ingroup program
 */
class StringLiteral: public Literal {
public:
  /**
   * Constructor.
   */
  StringLiteral();

  /**
   * Destructor.
   */
  virtual ~StringLiteral();
};
}

#endif

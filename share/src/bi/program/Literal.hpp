/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LITERAL_HPP
#define BI_PROGRAM_LITERAL_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Literal.
 *
 * @ingroup program
 */
class Literal: public Expression {
public:
  /**
   * Constructor.
   */
  Literal();

  /**
   * Destructor.
   */
  virtual ~Literal();
};
}

#endif

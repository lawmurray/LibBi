/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_UNARYEXPRESSION_HPP
#define BI_PROGRAM_UNARYEXPRESSION_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Unary expression.
 *
 * @ingroup program
 */
class UnaryExpression: public Expression {
public:
  /**
   * Constructor.
   */
  UnaryExpression();

  /**
   * Destructor.
   */
  virtual ~UnaryExpression();
};
}

#endif


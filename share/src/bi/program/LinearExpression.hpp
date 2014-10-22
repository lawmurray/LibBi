/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LINEAREXPRESSION_HPP
#define BI_PROGRAM_LINEAREXPRESSION_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Linear expression.
 *
 * @ingroup program
 */
class LinearExpression: public Expression {
public:
  /**
   * Constructor.
   */
  LinearExpression();

  /**
   * Destructor.
   */
  virtual ~LinearExpression();
};
}

#endif

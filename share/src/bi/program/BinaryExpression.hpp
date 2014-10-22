/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BINARYEXPRESSION_HPP
#define BI_PROGRAM_BINARYEXPRESSION_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class BinaryExpression: public Expression {
public:
  /**
   * Constructor.
   */
  BinaryExpression();

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();
};
}

#endif


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
#include "Operator.hpp"

#include "boost/shared_ptr.hpp"

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
  BinaryExpression(Expression* left, Operator* op, Expression* right);

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();

  /**
   * Left operand.
   */
  boost::shared_ptr<Expression> left;

  /**
   * Operator.
   */
  boost::shared_ptr<Operator> op;

  /**
   * Right operand.
   */
  boost::shared_ptr<Expression> right;
};
}

inline biprog::BinaryExpression::BinaryExpression(Expression* left,
    Operator* op, Expression* right) :
    left(left), op(op), right(right) {
  //
}

inline biprog::BinaryExpression::~BinaryExpression() {
  //
}

#endif

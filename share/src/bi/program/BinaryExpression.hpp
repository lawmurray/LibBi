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

#include "boost/scoped_ptr.hpp"

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
  BinaryExpression(Expression* left, Operator op, Expression* right);

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();

  virtual bool match(BinaryExpression* o, Match& match);

  /**
   * Left operand.
   */
  boost::scoped_ptr<Expression> left;

  /**
   * Operator.
   */
  Operator op;

  /**
   * Right operand.
   */
  boost::scoped_ptr<Expression> right;
};
}

inline biprog::BinaryExpression::BinaryExpression(Expression* left,
    Operator op, Expression* right) :
    left(left), op(op), right(right) {
  //
}

inline biprog::BinaryExpression::~BinaryExpression() {
  //
}

inline bool biprog::BinaryExpression::match(BinaryExpression* o, Match& match) {
  if (this->op)
  match.push(o, this, Match::SCORE_EXPRESSION);
  return true;
}

#endif

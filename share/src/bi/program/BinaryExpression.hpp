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

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class BinaryExpression: public Expression,
    public boost::enable_shared_from_this<BinaryExpression> {
public:
  /**
   * Constructor.
   */
  BinaryExpression(Expression* left, Operator op, Expression* right);

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();

  virtual bool match(boost::shared_ptr<BinaryExpression> o, Match& match);

  /**
   * Left operand.
   */
  boost::shared_ptr<Expression> left;

  /**
   * Operator.
   */
  Operator op;

  /**
   * Right operand.
   */
  boost::shared_ptr<Expression> right;
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

inline bool biprog::BinaryExpression::match(
    boost::shared_ptr<BinaryExpression> o, Match& match) {
  return op == o->op && left->match(o->left, match)
      && right->match(o->right, match);
}

#endif

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
#include "Operator.hpp"

namespace biprog {
/**
 * Unary expression.
 *
 * @ingroup program
 */
class UnaryExpression: public Expression,
    public boost::enable_shared_from_this<UnaryExpression> {
public:
  /**
   * Constructor.
   *
   * @param op Operator.
   * @param right Right operand.
   */
  UnaryExpression(Operator op, Expression* right);

  /**
   * Destructor.
   */
  virtual ~UnaryExpression();

  virtual bool match(boost::shared_ptr<UnaryExpression> o, Match& match);

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

inline biprog::UnaryExpression::UnaryExpression(Operator op,
    Expression* right) :
    op(op), right(right) {
  //
}

inline biprog::UnaryExpression::~UnaryExpression() {
  //
}

inline bool biprog::UnaryExpression::match(
    boost::shared_ptr<UnaryExpression> o, Match& match) {
  return op == o->op && right->match(o->right, match);
}

#endif

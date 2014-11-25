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
class BinaryExpression: public virtual Expression {
public:
  /**
   * Constructor.
   */
  BinaryExpression(Expression* left, Operator op, Expression* right);

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();

  virtual BinaryExpression* clone();
  virtual Expression* accept(Visitor& v);

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;

  /**
   * Left operand.
   */
  Expression* left;

  /**
   * Operator.
   */
  Operator op;

  /**
   * Right operand.
   */
  Expression* right;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::BinaryExpression::BinaryExpression(Expression* left, Operator op,
    Expression* right) :
    Expression(left->type->clone()), left(left), op(op), right(right) {
  /* pre-conditions */
  BI_ASSERT(left);
  BI_ASSERT(right);

  type = left->type;  //@todo Infer type properly
}

inline biprog::BinaryExpression::~BinaryExpression() {
  delete left;
  delete right;
}

#endif

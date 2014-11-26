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
class UnaryExpression: public virtual Expression {
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

  virtual UnaryExpression* clone();
  virtual Expression* acceptExpression(Visitor& v);

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;

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

inline biprog::UnaryExpression::UnaryExpression(Operator op, Expression* right) :
    Expression(right->type->clone()), op(op), right(right) {
  /* pre-condition */
  BI_ASSERT(right);

  type = right->type;  //@todo Infer type properly
}

inline biprog::UnaryExpression::~UnaryExpression() {
  delete right;
}

#endif

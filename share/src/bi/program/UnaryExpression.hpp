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
class UnaryExpression: public virtual Expression,
    public boost::enable_shared_from_this<UnaryExpression> {
public:
  /**
   * Constructor.
   *
   * @param op Operator.
   * @param right Right operand.
   */
  UnaryExpression(Operator op, boost::shared_ptr<Expression> right);

  /**
   * Destructor.
   */
  virtual ~UnaryExpression();

  /*
   * Operators.
   */
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

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
    boost::shared_ptr<Expression> right) :
    op(op), right(right) {
  //
}

inline biprog::UnaryExpression::~UnaryExpression() {
  //
}

#endif

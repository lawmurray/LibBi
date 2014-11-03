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
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const UnaryExpression& o) const;
  virtual bool operator<=(const UnaryExpression& o) const;
  virtual bool operator>(const UnaryExpression& o) const;
  virtual bool operator>=(const UnaryExpression& o) const;
  virtual bool operator==(const UnaryExpression& o) const;
  virtual bool operator!=(const UnaryExpression& o) const;

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

inline bool biprog::UnaryExpression::operator<(
    const UnaryExpression& o) const {
  return op == o.op && *right < *o.right;
}

inline bool biprog::UnaryExpression::operator<=(
    const UnaryExpression& o) const {
  return op == o.op && *right <= *o.right;
}

inline bool biprog::UnaryExpression::operator>(
    const UnaryExpression& o) const {
  return op == o.op && *right > *o.right;
}

inline bool biprog::UnaryExpression::operator>=(
    const UnaryExpression& o) const {
  return op == o.op && *right >= *o.right;
}

inline bool biprog::UnaryExpression::operator==(
    const UnaryExpression& o) const {
  return op == o.op && *right == *o.right;
}

inline bool biprog::UnaryExpression::operator!=(
    const UnaryExpression& o) const {
  return op != o.op || *right != *o.right;
}

#endif

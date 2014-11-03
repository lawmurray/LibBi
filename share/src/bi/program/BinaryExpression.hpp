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
class BinaryExpression: public virtual Expression,
    public boost::enable_shared_from_this<BinaryExpression> {
public:
  /**
   * Constructor.
   */
  BinaryExpression(boost::shared_ptr<Expression> left, Operator op,
      boost::shared_ptr<Expression> right);

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();

  /*
   * Operators.
   */
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const BinaryExpression& o) const;
  virtual bool operator<=(const BinaryExpression& o) const;
  virtual bool operator>(const BinaryExpression& o) const;
  virtual bool operator>=(const BinaryExpression& o) const;
  virtual bool operator==(const BinaryExpression& o) const;
  virtual bool operator!=(const BinaryExpression& o) const;

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

inline biprog::BinaryExpression::BinaryExpression(
    boost::shared_ptr<Expression> left, Operator op,
    boost::shared_ptr<Expression> right) :
    left(left), op(op), right(right) {
  //
}

inline biprog::BinaryExpression::~BinaryExpression() {
  //
}

inline bool biprog::BinaryExpression::operator<(
    const BinaryExpression& o) const {
  return op == o.op && *left < *o.left && *right < *o.right;
}

inline bool biprog::BinaryExpression::operator<=(
    const BinaryExpression& o) const {
  return op == o.op && *left <= *o.left && *right <= *o.right;
}

inline bool biprog::BinaryExpression::operator>(
    const BinaryExpression& o) const {
  return op == o.op && *left > *o.left && *right > *o.right;
}

inline bool biprog::BinaryExpression::operator>=(
    const BinaryExpression& o) const {
  return op == o.op && *left >= *o.left && *right >= *o.right;
}

inline bool biprog::BinaryExpression::operator==(
    const BinaryExpression& o) const {
  return op == o.op && *left == *o.left && *right == *o.right;
}

inline bool biprog::BinaryExpression::operator!=(
    const BinaryExpression& o) const {
  return op != o.op || *left != *o.left || *right != *o.right;
}

#endif

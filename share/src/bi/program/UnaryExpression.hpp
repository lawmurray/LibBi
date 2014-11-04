/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
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

inline bool biprog::UnaryExpression::operator<(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right < *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::UnaryExpression::operator<=(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right <= *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::UnaryExpression::operator>(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right > *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::UnaryExpression::operator>=(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right >= *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::UnaryExpression::operator==(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right == *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::UnaryExpression::operator!=(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op != expr.op || *right != *expr.right;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif

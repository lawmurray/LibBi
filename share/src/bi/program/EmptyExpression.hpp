/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EMPTYEXPRESSION_HPP
#define BI_PROGRAM_EMPTYEXPRESSION_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Empty expression.
 *
 * @ingroup program
 *
 * Used for empty brackets, parentheses or braces.
 */
class EmptyExpression: public virtual Expression,
    public boost::enable_shared_from_this<EmptyExpression> {
public:
  /**
   * Destructor.
   */
  virtual ~EmptyExpression();

  /*
   * Operators.
   */
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const EmptyExpression& o) const;
  virtual bool operator<=(const EmptyExpression& o) const;
  virtual bool operator>(const EmptyExpression& o) const;
  virtual bool operator>=(const EmptyExpression& o) const;
  virtual bool operator==(const EmptyExpression& o) const;
  virtual bool operator!=(const EmptyExpression& o) const;
};
}

inline biprog::EmptyExpression::~EmptyExpression() {
  //
}

inline bool biprog::EmptyExpression::operator<(
    const EmptyExpression& o) const {
  return false;
}

inline bool biprog::EmptyExpression::operator<=(
    const EmptyExpression& o) const {
  return operator==(o);
}

inline bool biprog::EmptyExpression::operator>(
    const EmptyExpression& o) const {
  return false;
}

inline bool biprog::EmptyExpression::operator>=(
    const EmptyExpression& o) const {
  return operator==(o);
}

inline bool biprog::EmptyExpression::operator==(
    const EmptyExpression& o) const {
  return true;
}

inline bool biprog::EmptyExpression::operator!=(
    const EmptyExpression& o) const {
  return false;
}

#endif

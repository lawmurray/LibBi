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
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;
};
}

inline biprog::EmptyExpression::~EmptyExpression() {
  //
}

inline bool biprog::EmptyExpression::operator<(const Expression& o) const {
  return false;
}

inline bool biprog::EmptyExpression::operator<=(const Expression& o) const {
  return operator==(o);
}

inline bool biprog::EmptyExpression::operator>(const Expression& o) const {
  return false;
}

inline bool biprog::EmptyExpression::operator>=(const Expression& o) const {
  return operator==(o);
}

inline bool biprog::EmptyExpression::operator==(const Expression& o) const {
  try {
    const EmptyExpression& expr = dynamic_cast<const EmptyExpression&>(o);
    return true;
  } catch (std::bad_cast e) {
    return true;
  }
}

inline bool biprog::EmptyExpression::operator!=(const Expression& o) const {
  try {
    const EmptyExpression& expr = dynamic_cast<const EmptyExpression&>(o);
    return false;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif

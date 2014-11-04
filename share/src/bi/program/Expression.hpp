/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "boost/shared_ptr.hpp"
#include "boost/enable_shared_from_this.hpp"

namespace biprog {
/**
 * Expression.
 *
 * @ingroup program
 */
class Expression: public boost::enable_shared_from_this<Expression> {
public:
  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /*
   * Operators, to compare expressions in terms of the partial order induced
   * by specialisation.
   */
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;
};
}

inline biprog::Expression::~Expression() {
  //
}

inline bool biprog::Expression::operator<(const Expression& o) const {
  return false;
}

inline bool biprog::Expression::operator<=(const Expression& o) const {
  return false;
}

inline bool biprog::Expression::operator>(const Expression& o) const {
  return false;
}

inline bool biprog::Expression::operator>=(const Expression& o) const {
  return false;
}

inline bool biprog::Expression::operator==(const Expression& o) const {
  return false;
}

inline bool biprog::Expression::operator!=(const Expression& o) const {
  return true;
}

#endif

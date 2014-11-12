/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
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
   * Comparison operators for comparing expressions in terms of
   * specialisation.
   */
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;

  /**
   * Output operator. Defers to output() for polymorphism.
   */
  friend std::ostream& operator<<(std::ostream& out, const Expression& expr) {
    expr.output(out);
    return out;
  }

protected:
  /**
   * Output to stream.
   */
  virtual void output(std::ostream& out) const = 0;
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

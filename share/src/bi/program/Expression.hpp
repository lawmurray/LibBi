/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "../misc/assert.hpp"

namespace biprog {
class Visitor;

/**
 * Expression.
 *
 * @ingroup program
 */
class Expression {
public:
  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /*
   * Bool cast to check for non-empty expression.
   */
  virtual operator bool() const;

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

#endif

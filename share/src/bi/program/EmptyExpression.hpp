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
class EmptyExpression: public virtual Expression {
public:
  /**
   * Destructor.
   */
  virtual ~EmptyExpression();

  virtual EmptyExpression* clone();
  virtual Expression* accept(Visitor& v);

  virtual operator bool() const;

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::EmptyExpression::~EmptyExpression() {
  //
}

#endif

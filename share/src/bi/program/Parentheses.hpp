/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PARENTHESES_HPP
#define BI_PROGRAM_PARENTHESES_HPP

#include "Grouped.hpp"

namespace biprog {
/**
 * Parentheses.
 *
 * @ingroup program
 */
class Parentheses: public virtual Grouped {
public:
  /**
   * Constructor.
   */
  Parentheses(Expression* expr);

  /**
   * Destructor.
   */
  virtual ~Parentheses();

  virtual Parentheses* clone();
  virtual Expression* acceptExpression(Visitor& v);

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Parentheses::Parentheses(Expression* expr) :
    Grouped(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Parentheses::~Parentheses() {
  //
}

#endif

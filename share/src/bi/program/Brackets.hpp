/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACKETS_HPP
#define BI_PROGRAM_BRACKETS_HPP

#include "Grouped.hpp"

namespace biprog {
/**
 * Square brackets.
 *
 * @ingroup program
 */
class Brackets: public virtual Grouped {
public:
  /**
   * Constructor.
   */
  Brackets(Expression* expr);

  /**
   * Destructor.
   */
  virtual ~Brackets();

  virtual Brackets* clone();
  virtual Expression* acceptExpression(Visitor& v);

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;

protected:
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Brackets::Brackets(Expression* expr) :
    Grouped(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Brackets::~Brackets() {
  //
}

#endif

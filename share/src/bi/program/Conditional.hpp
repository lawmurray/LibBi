/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONAL_HPP
#define BI_PROGRAM_CONDITIONAL_HPP

#include "Statement.hpp"
#include "Conditioned.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public virtual Statement,
    public virtual Conditioned,
    public virtual Braced {
public:
  /**
   * Constructor.
   */
  Conditional(Expression* cond, Statement* braces, Statement* falseBraces);

  /**
   * Destructor.
   */
  virtual ~Conditional();

  virtual Conditional* clone();
  virtual Statement* accept(Visitor& v);

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

  /**
   * Block if condition is false. May be empty if there is no else clause.
   */
  Statement* falseBraces;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Conditional::Conditional(Expression* cond, Statement* braces,
    Statement* falseBraces) :
    Conditioned(cond), Braced(braces), falseBraces(falseBraces) {
  /* pre-condition */
  BI_ASSERT(falseBraces);
}

inline biprog::Conditional::~Conditional() {
  delete falseBraces;
}

#endif

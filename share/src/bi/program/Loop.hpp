/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LOOP_HPP
#define BI_PROGRAM_LOOP_HPP

#include "Statement.hpp"
#include "Conditioned.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Loop.
 *
 * @ingroup program
 */
class Loop: public virtual Statement,
    public virtual Conditioned,
    public virtual Braced {
public:
  /**
   * Constructor.
   */
  Loop(Expression* cond, Expression* braces);

  /**
   * Destructor.
   */
  virtual ~Loop();

  virtual Loop* clone();
  virtual Statement* acceptStatement(Visitor& v);

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Loop::Loop(Expression* cond, Expression* braces) :
    Conditioned(cond), Braced(braces) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONAL_HPP
#define BI_PROGRAM_CONDITIONAL_HPP

#include "Typed.hpp"
#include "Conditioned.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public virtual Typed,
    public virtual Conditioned,
    public virtual Braced {
public:
  /**
   * Constructor.
   */
  Conditional(Typed* cond, Typed* braces, Typed* falseBraces);

  /**
   * Destructor.
   */
  virtual ~Conditional();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Block if condition is false. May be empty if there is no else clause.
   */
  Typed* falseBraces;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Conditional::Conditional(Typed* cond, Typed* braces,
    Typed* falseBraces) :
    Conditioned(cond), Braced(braces), falseBraces(falseBraces) {
  /* pre-condition */
  BI_ASSERT(falseBraces);
}

inline biprog::Conditional::~Conditional() {
  delete falseBraces;
}

#endif

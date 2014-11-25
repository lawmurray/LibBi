/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LOOP_HPP
#define BI_PROGRAM_LOOP_HPP

#include "Typed.hpp"
#include "Conditioned.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Loop.
 *
 * @ingroup program
 */
class Loop: public virtual Typed,
    public virtual Conditioned,
    public virtual Braced {
public:
  /**
   * Constructor.
   */
  Loop(Typed* cond, Typed* braces);

  /**
   * Destructor.
   */
  virtual ~Loop();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Loop::Loop(Typed* cond, Typed* braces) :
    Conditioned(cond), Braced(braces) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

#endif

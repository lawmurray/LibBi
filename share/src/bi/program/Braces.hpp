/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACES_HPP
#define BI_PROGRAM_BRACES_HPP

#include "Typed.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Braces.
 *
 * @ingroup program
 */
class Braces: public virtual Typed, public virtual Scoped {
public:
  /**
   * Constructor.
   */
  Braces();

  /**
   * Constructor.
   */
  Braces(Typed* expr);

  /**
   * Destructor.
   */
  virtual ~Braces();

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

inline biprog::Braces::Braces() {
  //
}

inline biprog::Braces::Braces(Typed* expr) :
    Typed(expr->type->clone()), Scoped(expr) {
  //
}

inline biprog::Braces::~Braces() {
  //
}

#endif

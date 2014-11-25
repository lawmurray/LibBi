/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_REFERENCE_HPP
#define BI_PROGRAM_REFERENCE_HPP

#include "Named.hpp"
#include "Bracketed.hpp"
#include "Parenthesised.hpp"
#include "Typed.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class Reference: public virtual Named,
    public virtual Bracketed,
    public virtual Parenthesised,
    public virtual Typed,
    public virtual Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param brackets Expression in square brackets.
   * @param parens Expression in parentheses.
   * @param type Type.
   * @param braces Expression in braces.
   * @param target Target of the reference. May be null if unresolved.
   */
  Reference(const std::string name, Typed* brackets, Typed* parens,
      Typed* type, Typed* braces, Typed* target = NULL);

  /**
   * Destructor.
   */
  virtual ~Reference();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Target.
   */
  Typed* target;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Reference::Reference(const std::string name, Typed* brackets,
    Typed* parens, Typed* type, Typed* braces, Typed* target) :
    Named(name), Bracketed(brackets), Parenthesised(parens), Typed(type), Braced(
        braces), target(target) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif


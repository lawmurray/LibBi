/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DEF_HPP
#define BI_PROGRAM_DEF_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Typed.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Def.
 *
 * @ingroup program
 */
class Def: public virtual Named,
    public virtual Parenthesised,
    public virtual Typed,
    public virtual Braced {
public:
  /**
   * Constructor.
   */
  Def(const std::string name, Typed* parens, Typed* type, Typed* braces);

  /**
   * Destructor.
   */
  virtual ~Def();

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

inline biprog::Def::Def(const std::string name, Typed* parens, Typed* type,
    Typed* braces) :
    Named(name), Parenthesised(parens), Typed(type), Braced(braces) {
  //
}

inline biprog::Def::~Def() {
  //
}

#endif

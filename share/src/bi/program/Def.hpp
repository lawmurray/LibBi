/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DEF_HPP
#define BI_PROGRAM_DEF_HPP

#include "Statement.hpp"
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
class Def: public virtual Statement,
    public virtual Named,
    public virtual Typed,
    public virtual Parenthesised,
    public virtual Braced {
public:
  /**
   * Constructor.
   */
  Def(const std::string name, Expression* parens, Statement* type,
      Expression* braces);

  /**
   * Destructor.
   */
  virtual ~Def();

  virtual Def* clone();
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

inline biprog::Def::Def(const std::string name, Expression* parens,
    Statement* type, Expression* braces) :
    Named(name), Parenthesised(parens), Typed(type), Braced(braces) {
  //
}

inline biprog::Def::~Def() {
  //
}

#endif

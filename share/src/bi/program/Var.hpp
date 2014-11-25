/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_VAR_HPP
#define BI_PROGRAM_VAR_HPP

#include "Statement.hpp"
#include "Named.hpp"
#include "Bracketed.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public virtual Statement,
    public virtual Named,
    public virtual Typed,
    public virtual Bracketed {
public:
  /**
   * Constructor.
   */
  Var(const std::string name, Expression* brackets, Expression* type);

  /**
   * Destructor.
   */
  virtual ~Var();

  virtual Var* clone();
  virtual Statement* accept(Visitor& v);

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Var::Var(const std::string name, Expression* brackets,
    Expression* type) :
    Named(name), Bracketed(brackets), Typed(type) {
  //
}

inline biprog::Var::~Var() {
  //
}

#endif

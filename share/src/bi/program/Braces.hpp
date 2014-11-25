/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACES_HPP
#define BI_PROGRAM_BRACES_HPP

#include "Statement.hpp"
#include "Expression.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Braces.
 *
 * @ingroup program
 */
class Braces: public virtual Statement,
    public virtual Expression,
    public virtual Scoped {
public:
  /**
   * Constructor.
   */
  Braces();

  /**
   * Constructor.
   */
  Braces(Statement* expr);

  /**
   * Destructor.
   */
  virtual ~Braces();

  virtual Braces* clone();
  virtual Braces* accept(Visitor& v);

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

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

inline biprog::Braces::Braces(Statement* stmt) :
    Expression(NULL), Scoped(stmt) {
  //
}

inline biprog::Braces::~Braces() {
  //
}

#endif

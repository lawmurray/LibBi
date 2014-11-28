/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACES_HPP
#define BI_PROGRAM_BRACES_HPP

#include "Expression.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Braces.
 *
 * @ingroup program
 */
class Braces: public virtual Expression, public virtual Scoped {
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
  virtual Expression* acceptExpression(Visitor& v);

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;

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
    Scoped(stmt) {
  //
}

inline biprog::Braces::~Braces() {
  //
}

#endif

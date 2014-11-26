/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_REFERENCE_HPP
#define BI_PROGRAM_REFERENCE_HPP

#include "Statement.hpp"
#include "Expression.hpp"
#include "Named.hpp"
#include "Bracketed.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"
#include "EmptyExpression.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class Reference: public virtual Statement,
    public virtual Expression,
    public virtual Named,
    public virtual Bracketed,
    public virtual Parenthesised,
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
  Reference(const std::string name, Expression* brackets, Expression* parens,
      Statement* type, Expression* braces, Expression* target = new EmptyExpression());

  /**
   * Destructor.
   */
  virtual ~Reference();

  virtual Reference* clone();
  virtual Expression* acceptExpression(Visitor& v);
  virtual Statement* acceptStatement(Visitor& v);

  virtual bool operator<=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

  /**
   * Target.
   */
  Expression* target;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Reference::Reference(const std::string name,
    Expression* brackets, Expression* parens, Statement* type,
    Expression* braces, Expression* target) :
    Named(name), Bracketed(brackets), Parenthesised(parens), Expression(type), Braced(
        braces), target(target) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif


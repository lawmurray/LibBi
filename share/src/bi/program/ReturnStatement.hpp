/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_RETURNSTATEMENT_HPP
#define BI_PROGRAM_RETURNSTATEMENT_HPP

#include "Statement.hpp"
#include "Expression.hpp"

namespace biprog {
/**
 * Return statement.
 *
 * @ingroup program
 */
class ReturnStatement: public virtual Statement {
public:
  /**
   * Constructor.
   *
   * @param expr Expression.
   */
  ReturnStatement(Expression* expr);

  /**
   * Destructor.
   */
  virtual ~ReturnStatement();

  virtual ReturnStatement* clone();
  virtual Statement* accept(Visitor& v);

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

  /**
   * Right operand.
   */
  Expression* expr;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::ReturnStatement::ReturnStatement(Expression* expr) :
    expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::ReturnStatement::~ReturnStatement() {
  //
}

#endif

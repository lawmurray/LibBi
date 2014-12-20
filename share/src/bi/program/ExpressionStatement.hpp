/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSIONSTATEMENT_HPP
#define BI_PROGRAM_EXPRESSIONSTATEMENT_HPP

#include "Statement.hpp"
#include "Expression.hpp"

namespace biprog {
/**
 * Expression statement.
 *
 * @ingroup program
 */
class ExpressionStatement: public virtual Statement {
public:
  /**
   * Constructor.
   *
   * @param expr Expression.
   */
  ExpressionStatement(Expression* expr);

  /**
   * Destructor.
   */
  virtual ~ExpressionStatement();

  virtual ExpressionStatement* clone();
  virtual Statement* acceptStatement(Visitor& v);

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

inline biprog::ExpressionStatement::ExpressionStatement(Expression* expr) :
    expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::ExpressionStatement::~ExpressionStatement() {
  //
}

#endif

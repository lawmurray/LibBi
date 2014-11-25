/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_RETURNEXPRESSION_HPP
#define BI_PROGRAM_RETURNEXPRESSION_HPP

#include "Typed.hpp"

namespace biprog {
/**
 * Unary expression.
 *
 * @ingroup program
 */
class ReturnExpression: public virtual Typed {
public:
  /**
   * Constructor.
   *
   * @param expr Right operand.
   */
  ReturnExpression(Typed* expr);

  /**
   * Destructor.
   */
  virtual ~ReturnExpression();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Right operand.
   */
  Typed* expr;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::ReturnExpression::ReturnExpression(Typed* expr) :
    Typed(expr->type->clone()), expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::ReturnExpression::~ReturnExpression() {
  //
}

#endif

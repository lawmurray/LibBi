/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_GROUPED_HPP
#define BI_PROGRAM_GROUPED_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Grouped expression.
 *
 * @ingroup program
 */
class Grouped: public virtual Expression {
public:
  /**
   * Constructor.
   */
  Grouped();

  /**
   * Constructor.
   */
  Grouped(Expression* expr);

  /**
   * Destructor.
   */
  virtual ~Grouped();

  /**
   * Grouped expression.
   */
  Expression* expr;
};
}

inline biprog::Grouped::Grouped() :
    expr(NULL) {
  //
}

inline biprog::Grouped::Grouped(Expression* expr) :
    Expression(expr->type->clone()), expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Grouped::~Grouped() {
  delete expr;
}

#endif

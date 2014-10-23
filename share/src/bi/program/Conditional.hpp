/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONAL_HPP
#define BI_PROGRAM_CONDITIONAL_HPP

#include "Statement.hpp"
#include "Expression.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public Statement {
public:
  /**
   * Constructor.
   */
  Conditional(Expression* cond, Statement* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Conditional();

  /**
   * Condition.
   */
  Expression* cond;

  /**
   * Body.
   */
  Statement* body;
};
}

inline biprog::Conditional::Conditional(Expression* cond, Statement* body) :
    cond(cond), body(body) {
  //
}

inline biprog::Conditional::~Conditional() {
  //
}

#endif

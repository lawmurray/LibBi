/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LOOP_HPP
#define BI_PROGRAM_LOOP_HPP

#include "Statement.hpp"
#include "Expression.hpp"

namespace biprog {
/**
 * Loop.
 *
 * @ingroup program
 */
class Loop: public Statement {
public:
  /**
   * Constructor.
   */
  Loop(Expression* cond, Statement* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Loop();

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

inline biprog::Loop::Loop(Expression* cond, Statement* body) :
    cond(cond), body(body) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

#endif

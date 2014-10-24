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

#include "boost/shared_ptr.hpp"

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
  Loop(Expression* cond, Expression* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Loop();

  /**
   * Condition.
   */
  boost::shared_ptr<Expression> cond;

  /**
   * Body.
   */
  boost::shared_ptr<Expression> body;
};
}

inline biprog::Loop::Loop(Expression* cond, Expression* body) :
    cond(cond), body(body) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

#endif

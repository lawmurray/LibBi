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
#include "Conditioned.hpp"
#include "Bodied.hpp"
#include "Expression.hpp"

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Loop.
 *
 * @ingroup program
 */
class Loop: public Statement, public Conditioned, public Bodied {
public:
  /**
   * Constructor.
   */
  Loop(Expression* cond, Expression* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Loop();
};
}

inline biprog::Loop::Loop(Expression* cond, Expression* body) :
    Conditioned(cond), Bodied(body) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

#endif

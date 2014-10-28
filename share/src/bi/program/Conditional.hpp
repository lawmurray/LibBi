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
#include "Conditioned.hpp"
#include "Bodied.hpp"
#include "Expression.hpp"

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public Statement, public Conditioned, public Bodied {
public:
  /**
   * Constructor.
   */
  Conditional(Expression* cond, Expression* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Conditional();
};
}

inline biprog::Conditional::Conditional(Expression* cond, Expression* body) :
    Conditioned(cond), Bodied(body) {
  //
}

inline biprog::Conditional::~Conditional() {
  //
}

#endif

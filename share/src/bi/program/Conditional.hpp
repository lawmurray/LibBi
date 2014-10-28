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
#include "Braced.hpp"
#include "Expression.hpp"

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public Statement, public Conditioned, public Braced {
public:
  /**
   * Constructor.
   */
  Conditional(Expression* cond, Expression* braces = NULL);

  /**
   * Destructor.
   */
  virtual ~Conditional();
};
}

inline biprog::Conditional::Conditional(Expression* cond, Expression* braces) :
    Conditioned(cond), Braced(braces) {
  //
}

inline biprog::Conditional::~Conditional() {
  //
}

#endif

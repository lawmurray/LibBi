/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONED_HPP
#define BI_PROGRAM_CONDITIONED_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Conditioned expression.
 *
 * @ingroup program
 */
class Conditioned : public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param cond Conditional expression.
   */
  Conditioned(boost::shared_ptr<Expression> cond);

  /**
   * Destructor.
   */
  virtual ~Conditioned() = 0;

protected:
  /**
   * First statement in in brackets.
   */
  boost::shared_ptr<Expression> cond;
};
}

inline biprog::Conditioned::Conditioned(boost::shared_ptr<Expression> cond) :
    cond(cond) {
  //
}

inline biprog::Conditioned::~Conditioned() {
  //
}

#endif


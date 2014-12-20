/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
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
class Conditioned {
public:
  /**
   * Constructor.
   *
   * @param cond Conditional expression.
   */
  Conditioned(Expression* cond);

  /**
   * Destructor.
   */
  virtual ~Conditioned() = 0;

  /**
   * First statement in in brackets.
   */
  Expression* cond;
};
}

inline biprog::Conditioned::Conditioned(Expression* cond) :
    cond(cond) {
  /* pre-condition */
  BI_ASSERT(cond);
}

inline biprog::Conditioned::~Conditioned() {
  delete cond;
}

#endif


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
class Conditioned: public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param cond Conditional expression.
   */
  Conditioned(Typed* cond);

  /**
   * Destructor.
   */
  virtual ~Conditioned() = 0;

  /**
   * First statement in in brackets.
   */
  Typed* cond;
};
}

inline biprog::Conditioned::Conditioned(Typed* cond) :
    cond(cond) {
  /* pre-condition */
  BI_ASSERT(cond);
}

inline biprog::Conditioned::~Conditioned() {
  delete cond;
}

#endif


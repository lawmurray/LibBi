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
class Conditioned : public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param cond Conditional expression.
   */
  Conditioned(boost::shared_ptr<Typed> cond);

  /**
   * Destructor.
   */
  virtual ~Conditioned() = 0;

protected:
  /**
   * First statement in in brackets.
   */
  boost::shared_ptr<Typed> cond;
};
}

inline biprog::Conditioned::Conditioned(boost::shared_ptr<Typed> cond) :
    cond(cond) {
  /* pre-condition */
  BI_ASSERT(cond);
}

inline biprog::Conditioned::~Conditioned() {
  //
}

#endif


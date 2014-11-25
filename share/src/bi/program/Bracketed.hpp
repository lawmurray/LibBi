/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACKETED_HPP
#define BI_PROGRAM_BRACKETED_HPP

#include "Expression.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Bracketed expression.
 *
 * @ingroup program
 */
class Bracketed : public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param brackets Expression in square brackets.
   */
  Bracketed(Typed* brackets);

  /**
   * Destructor.
   */
  virtual ~Bracketed() = 0;

  /**
   * First statement in index brackets.
   */
  Typed* brackets;
};
}

inline biprog::Bracketed::Bracketed(Typed* brackets) :
    brackets(brackets) {
  /* pre-condition */
  BI_ASSERT(brackets);
}

inline biprog::Bracketed::~Bracketed() {
  delete brackets;
}

#endif


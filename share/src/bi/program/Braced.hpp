/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACED_HPP
#define BI_PROGRAM_BRACED_HPP

#include "Expression.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Braced expression.
 *
 * @ingroup program
 */
class Braced: public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param in Expression in curly brackets.
   */
  Braced(Typed* braces);

  /**
   * Destructor.
   */
  virtual ~Braced() = 0;

  /**
   * First statement in in brackets.
   */
  Typed* braces;
};
}

inline biprog::Braced::Braced(Typed* braces) :
    braces(braces) {
  /* pre-condition */
  BI_ASSERT(braces);
}

inline biprog::Braced::~Braced() {
  delete braces;
}

#endif


/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACED_HPP
#define BI_PROGRAM_BRACED_HPP

#include "Statement.hpp"

namespace biprog {
/**
 * Braced expression.
 *
 * @ingroup program
 */
class Braced {
public:
  /**
   * Constructor.
   *
   * @param in Expression in curly brackets.
   */
  Braced(Statement* braces);

  /**
   * Destructor.
   */
  virtual ~Braced() = 0;

  /**
   * First statement in in brackets.
   */
  Statement* braces;
};
}

inline biprog::Braced::Braced(Statement* braces) :
    braces(braces) {
  /* pre-condition */
  BI_ASSERT(braces);
}

inline biprog::Braced::~Braced() {
  delete braces;
}

#endif


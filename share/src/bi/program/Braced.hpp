/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACED_HPP
#define BI_PROGRAM_BRACED_HPP

#include "boost/scoped_ptr.hpp"

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
   * @param in Statement in curly brackets.
   */
  Braced(Expression* braces = NULL);

  /**
   * Destructor.
   */
  virtual ~Braced() = 0;

  /**
   * First statement in in brackets.
   */
  boost::scoped_ptr<Expression> braces;
};
}

inline biprog::Braced::Braced(Expression* braces) :
    braces(braces) {
  //
}

inline biprog::Braced::~Braced() {
  //
}

#endif


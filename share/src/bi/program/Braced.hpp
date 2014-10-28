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
  Braced(boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Braced() = 0;

  /**
   * First statement in in brackets.
   */
  boost::shared_ptr<Expression> braces;
};
}

inline biprog::Braced::Braced(boost::shared_ptr<Expression> braces) :
    braces(braces) {
  //
}

inline biprog::Braced::~Braced() {
  //
}

#endif


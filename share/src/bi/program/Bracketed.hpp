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

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Bracketed expression.
 *
 * @ingroup program
 */
class Bracketed {
public:
  /**
   * Constructor.
   *
   * @param brackets Expression in square brackets.
   */
  Bracketed(Expression* brackets = NULL);

  /**
   * Destructor.
   */
  virtual ~Bracketed() = 0;

  /**
   * First statement in index brackets.
   */
  boost::scoped_ptr<Expression> brackets;
};
}

inline biprog::Bracketed::Bracketed(Expression* brackets) :
    brackets(brackets) {
  //
}

inline biprog::Bracketed::~Bracketed() {
  //
}

#endif


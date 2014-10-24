/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PARENTHESISED_HPP
#define BI_PROGRAM_PARENTHESISED_HPP

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Parenthesised expression.
 *
 * @ingroup program
 */
class Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param in Expression in parentheses.
   */
  Parenthesised(Expression* in = NULL);

  /**
   * Destructor.
   */
  virtual ~Parenthesised() = 0;

  /**
   * First statement in in brackets.
   */
  boost::scoped_ptr<Expression> in;
};
}

inline biprog::Parenthesised::Parenthesised(Expression* in) :
    in(in) {
  //
}

inline biprog::Parenthesised::~Parenthesised() {
  //
}

#endif


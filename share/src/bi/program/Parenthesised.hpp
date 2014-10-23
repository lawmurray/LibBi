/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PARENTHESISED_HPP
#define BI_PROGRAM_PARENTHESISED_HPP

#include "boost/shared_ptr.hpp"

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
   * @param in Statement in parentheses.
   */
  Parenthesised(Statement* in = NULL);

  /**
   * Destructor.
   */
  virtual ~Parenthesised() = 0;

  /**
   * First statement in in brackets.
   */
  boost::shared_ptr<Statement> in;
};
}

inline biprog::Parenthesised::Parenthesised(Statement* in) :
    in(in) {
  //
}

inline biprog::Parenthesised::~Parenthesised() {
  //
}

#endif


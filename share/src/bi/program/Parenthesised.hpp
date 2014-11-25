/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PARENTHESISED_HPP
#define BI_PROGRAM_PARENTHESISED_HPP

#include "Expression.hpp"

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
   * @param parens Expression in parentheses.
   */
  Parenthesised(Expression* parens);

  /**
   * Destructor.
   */
  virtual ~Parenthesised() = 0;

  /**
   * First statement in in brackets.
   */
  Expression* parens;
};
}

inline biprog::Parenthesised::Parenthesised(Expression* parens) :
    parens(parens) {
  /* pre-condition */
  BI_ASSERT(parens);
}

inline biprog::Parenthesised::~Parenthesised() {
  delete parens;
}

#endif


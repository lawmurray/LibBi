/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "Match.hpp"

namespace biprog {
/**
 * Expression.
 *
 * @ingroup program
 */
class Expression {
public:

  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /**
   * Attempt to match another expression.
   *
   * @param o Other expression.
   * @param[out] match The match, if any.
   *
   * @return Does #o match?
   */
  virtual bool match(Expression* o, Match& match);
};
}

inline biprog::Expression::~Expression() {
  //
}

inline bool biprog::Expression::match(Expression* o, Match& match) {
  match.push(o, this, Match::SCORE_EXPRESSION);
}

#endif


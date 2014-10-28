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

#include "boost/shared_ptr.hpp"
#include "boost/enable_shared_from_this.hpp"

namespace biprog {
/**
 * Expression.
 *
 * @ingroup program
 */
class Expression: public boost::enable_shared_from_this<Expression> {
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
  virtual bool match(boost::shared_ptr<Expression> o, Match& match);
};
}

inline biprog::Expression::~Expression() {
  //
}

inline bool biprog::Expression::match(boost::shared_ptr<Expression> o,
    Match& match) {
  return o && match.push(o, shared_from_this(), Match::SCORE_EXPRESSION);
}

#endif


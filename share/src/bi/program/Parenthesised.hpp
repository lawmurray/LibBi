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
#include "Typed.hpp"

namespace biprog {
/**
 * Parenthesised expression.
 *
 * @ingroup program
 */
class Parenthesised : public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param parens Expression in parentheses.
   */
  Parenthesised(boost::shared_ptr<Typed> parens);

  /**
   * Destructor.
   */
  virtual ~Parenthesised() = 0;

  /**
   * First statement in in brackets.
   */
  boost::shared_ptr<Typed> parens;
};
}

inline biprog::Parenthesised::Parenthesised(boost::shared_ptr<Typed> parens) :
    parens(parens) {
  /* pre-condition */
  BI_ASSERT(parens);
}

inline biprog::Parenthesised::~Parenthesised() {
  //
}

#endif


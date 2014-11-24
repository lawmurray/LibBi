/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_GROUPED_HPP
#define BI_PROGRAM_GROUPED_HPP

#include "Typed.hpp"

namespace biprog {
/**
 * Grouped expression.
 *
 * @ingroup program
 */
class Grouped: public virtual Expression {
public:
  /**
   * Constructor.
   */
  Grouped();

  /**
   * Constructor.
   */
  Grouped(boost::shared_ptr<Typed> expr);

  /**
   * Destructor.
   */
  virtual ~Grouped();

  /**
   * Grouped expression.
   */
  boost::shared_ptr<Typed> expr;
};
}

#include "EmptyExpression.hpp"

inline biprog::Grouped::Grouped() : expr(boost::make_shared<EmptyExpression>()) {
  //
}

inline biprog::Grouped::Grouped(boost::shared_ptr<Typed> expr) :
    expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Grouped::~Grouped() {
  //
}

#endif

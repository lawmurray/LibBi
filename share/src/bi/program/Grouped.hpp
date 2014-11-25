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
  Grouped(Typed* expr);

  /**
   * Destructor.
   */
  virtual ~Grouped();

  /**
   * Grouped expression.
   */
  Typed* expr;
};
}

inline biprog::Grouped::Grouped() :
    expr(NULL) {
  //
}

inline biprog::Grouped::Grouped(Typed* expr) :
    expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Grouped::~Grouped() {
  delete expr;
}

#endif

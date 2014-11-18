/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DERIVED_HPP
#define BI_PROGRAM_DERIVED_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Derived type.
 *
 * @ingroup program
 */
class Derived: public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param base Base type.
   */
  Derived(boost::shared_ptr<Expression> base);

  /**
   * Destructor.
   */
  virtual ~Derived() = 0;

protected:
  /**
   * Base type.
   */
  boost::shared_ptr<Expression> base;
};
}

inline biprog::Derived::Derived(boost::shared_ptr<Expression> base) :
    base(base) {
  //
}

inline biprog::Derived::~Derived() {
  //
}

#endif

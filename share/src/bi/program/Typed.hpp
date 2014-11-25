/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_TYPED_HPP
#define BI_PROGRAM_TYPED_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Typed declaration.
 *
 * @ingroup program
 */
class Typed {
public:
  /**
   * Constructor.
   *
   * @param type Type.
   */
  Typed(Expression* type);

  /**
   * Destructor.
   */
  virtual ~Typed() = 0;

  /**
   * Type.
   */
  Expression* type;
};
}

inline biprog::Typed::Typed(Expression* type) :
    type(type) {
  //
}

inline biprog::Typed::~Typed() {
  delete type;
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_TYPED_HPP
#define BI_PROGRAM_TYPED_HPP

#include "Statement.hpp"
#include "EmptyStatement.hpp"

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
  Typed(Statement* type = new EmptyStatement());

  /**
   * Destructor.
   */
  virtual ~Typed() = 0;

  /**
   * Type.
   */
  Statement* type;
};
}

inline biprog::Typed::Typed(Statement* type) :
    type(type) {
  //
}

inline biprog::Typed::~Typed() {
  delete type;
}

#endif

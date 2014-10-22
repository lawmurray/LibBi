/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_INTLITERAL_HPP
#define BI_PROGRAM_INTLITERAL_HPP

#include "Literal.hpp"

namespace biprog {
/**
 * Int literal.
 *
 * @ingroup program
 */
class IntLiteral: public Literal<int> {
public:
  /**
   * Constructor.
   */
  IntLiteral(const int value);

  /**
   * Destructor.
   */
  virtual ~IntLiteral();
};
}

inline biprog::IntLiteral::IntLiteral(const int value) :
    Literal<int>(value) {
  //
}

inline biprog::IntLiteral::~IntLiteral() {
  //
}

#endif

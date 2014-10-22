/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BOOLLITERAL_HPP
#define BI_PROGRAM_BOOLLITERAL_HPP

#include "Literal.hpp"

namespace biprog {
/**
 * Bool literal.
 *
 * @ingroup program
 */
class BoolLiteral: public Literal<bool> {
public:
  /**
   * Constructor.
   */
  BoolLiteral(const bool value);

  /**
   * Destructor.
   */
  virtual ~BoolLiteral();
};
}

inline biprog::BoolLiteral::BoolLiteral(const bool value) :
    Literal<bool>(value) {
  //
}

inline biprog::BoolLiteral::~BoolLiteral() {
  //
}

#endif

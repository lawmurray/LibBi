/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DOUBLELITERAL_HPP
#define BI_PROGRAM_DOUBLELITERAL_HPP

#include "Literal.hpp"

namespace biprog {
/**
 * Double literal.
 *
 * @ingroup program
 */
class DoubleLiteral: public Literal<double> {
public:
  /**
   * Constructor.
   */
  DoubleLiteral(const double value);

  /**
   * Destructor.
   */
  virtual ~DoubleLiteral();
};
}

inline biprog::DoubleLiteral::DoubleLiteral(const double value) :
    Literal<double>(value) {
  //
}

inline biprog::DoubleLiteral::~DoubleLiteral() {
  //
}

#endif

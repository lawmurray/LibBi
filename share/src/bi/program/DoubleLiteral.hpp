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
class DoubleLiteral: public Literal {
public:
  /**
   * Constructor.
   */
  DoubleLiteral();

  /**
   * Destructor.
   */
  virtual ~DoubleLiteral();
};
}

#endif

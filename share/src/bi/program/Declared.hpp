/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DECLARED_HPP
#define BI_PROGRAM_DECLARED_HPP

#include "Statement.hpp"

namespace biprog {
/**
 * Declared.
 *
 * @ingroup program
 */
class Declared: public Statement {
public:
  /**
   * Constructor.
   */
  Declared();

  /**
   * Destructor.
   */
  virtual ~Declared();
};
}

#endif


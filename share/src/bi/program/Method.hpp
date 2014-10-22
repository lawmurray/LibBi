/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_METHOD_HPP
#define BI_PROGRAM_METHOD_HPP

#include "Declared.hpp"

namespace biprog {
/**
 * Method.
 *
 * @ingroup program
 */
class Method: public Declared {
public:
  /**
   * Constructor.
   */
  Method();

  /**
   * Destructor.
   */
  virtual ~Method();
};
}

#endif


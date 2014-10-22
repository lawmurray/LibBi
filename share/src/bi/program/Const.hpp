/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONST_HPP
#define BI_PROGRAM_CONST_HPP

#include "Declared.hpp"

namespace biprog {
/**
 * Constant.
 *
 * @ingroup program
 */
class Const: public Declared {
public:
  /**
   * Constructor.
   */
  Const();

  /**
   * Destructor.
   */
  virtual ~Const();
};
}

#endif

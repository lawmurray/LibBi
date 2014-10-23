/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONST_HPP
#define BI_PROGRAM_CONST_HPP

#include "Declaration.hpp"

namespace biprog {
/**
 * Constant.
 *
 * @ingroup program
 */
class Const: public Declaration {
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

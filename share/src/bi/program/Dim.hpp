/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DIM_HPP
#define BI_PROGRAM_DIM_HPP

#include "Declared.hpp"

namespace biprog {
/**
 * Dimension.
 *
 * @ingroup program
 */
class Dim: public Declared {
public:
  /**
   * Constructor.
   */
  Dim();

  /**
   * Destructor.
   */
  virtual ~Dim();
};
}

#endif

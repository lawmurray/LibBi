/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_MODEL_HPP
#define BI_PROGRAM_MODEL_HPP

#include "Declared.hpp"

namespace biprog {
/**
 * Model.
 *
 * @ingroup program
 */
class Model: public Declared {
public:
  /**
   * Constructor.
   */
  Model();

  /**
   * Destructor.
   */
  virtual ~Model();
};
}

#endif


/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DIM_HPP
#define BI_PROGRAM_DIM_HPP

#include "Declaration.hpp"

namespace biprog {
/**
 * Dimension.
 *
 * @ingroup program
 */
class Dim: public Declaration {
public:
  /**
   * Constructor.
   */
  Dim(Reference* ref);

  /**
   * Destructor.
   */
  virtual ~Dim();
};
}

inline biprog::Dim::Dim(Reference* ref) : Declaration(ref) {
  //
}

inline biprog::Dim::~Dim() {
  //
}

#endif

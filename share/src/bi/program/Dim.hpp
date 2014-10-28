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
#include "Bracketed.hpp"

namespace biprog {
/**
 * Dimension.
 *
 * @ingroup program
 */
class Dim: public Declaration, public Bracketed {
public:
  /**
   * Constructor.
   */
  Dim(const char* name, Expression* brackets);

  /**
   * Destructor.
   */
  virtual ~Dim();
};
}

inline biprog::Dim::Dim(const char* name, Expression* brackets) :
    Declaration(name), Bracketed(brackets) {
  //
}

inline biprog::Dim::~Dim() {
  //
}

#endif

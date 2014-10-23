/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_VAR_HPP
#define BI_PROGRAM_VAR_HPP

#include "Declaration.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public Declaration {
public:
  /**
   * Constructor.
   */
  Var(Reference* ref);

  /**
   * Destructor.
   */
  virtual ~Var();
};
}

inline biprog::Var::Var(Reference* ref) : Declaration(ref) {
  //
}

inline biprog::Var::~Var() {
  //
}

#endif

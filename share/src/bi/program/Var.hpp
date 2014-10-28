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
#include "Bracketed.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public Declaration, public Bracketed, public Typed {
public:
  /**
   * Constructor.
   */
  Var(const char* name, Expression* index, Type* type);

  /**
   * Destructor.
   */
  virtual ~Var();
};
}

inline biprog::Var::Var(const char* name, Expression* index, Type* type) :
    Declaration(name), Bracketed(index), Typed(type) {
  //
}

inline biprog::Var::~Var() {
  //
}

#endif

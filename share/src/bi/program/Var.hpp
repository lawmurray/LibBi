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
#include "Named.hpp"
#include "Typed.hpp"
#include "Bracketed.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public Declaration, public Named, public Typed, public Bracketed {
public:
  /**
   * Constructor.
   */
  Var(const char* name, Type* type, Statement* index = NULL);

  /**
   * Destructor.
   */
  virtual ~Var();
};
}

inline biprog::Var::Var(const char* name, Type* type, Statement* index) :
    Named(name), Typed(type), Bracketed(index) {
  //
}

inline biprog::Var::~Var() {
  //
}

#endif

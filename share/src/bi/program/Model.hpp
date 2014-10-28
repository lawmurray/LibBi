/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_MODEL_HPP
#define BI_PROGRAM_MODEL_HPP

#include "Declaration.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Model.
 *
 * @ingroup program
 */
class Model: public Declaration, public Parenthesised, public Braced {
public:
  /**
   * Constructor.
   */
  Model(const char* name, Expression* parens = NULL,
      Expression* braces = NULL);

  /**
   * Destructor.
   */
  virtual ~Model();
};
}

inline biprog::Model::Model(const char* name, Expression* parens,
    Expression* braces) :
    Declaration(name), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::Model::~Model() {
  //
}

#endif


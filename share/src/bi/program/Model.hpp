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
#include "Bodied.hpp"

namespace biprog {
/**
 * Model.
 *
 * @ingroup program
 */
class Model: public Declaration, public Parenthesised, public Bodied {
public:
  /**
   * Constructor.
   */
  Model(const char* name, Expression* in = NULL, Expression* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Model();
};
}

inline biprog::Model::Model(const char* name, Expression* in,
    Expression* body) :
    Declaration(name), Parenthesised(in), Bodied(body) {
  //
}

inline biprog::Model::~Model() {
  //
}

#endif


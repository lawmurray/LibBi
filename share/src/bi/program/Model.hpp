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
#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Bodied.hpp"

namespace biprog {
/**
 * Model.
 *
 * @ingroup program
 */
class Model: public Declaration,
    public Named,
    public Parenthesised,
    public Bodied {
public:
  /**
   * Constructor.
   */
  Model(const char* name, Statement* in = NULL, Statement* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Model();
};
}

inline biprog::Model::Model(const char* name, Statement* in, Statement* body) :
    Named(name), Parenthesised(in), Bodied(body) {
  //
}

inline biprog::Model::~Model() {
  //
}

#endif


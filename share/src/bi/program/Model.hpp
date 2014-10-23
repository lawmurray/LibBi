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

namespace biprog {
/**
 * Model.
 *
 * @ingroup program
 */
class Model: public Declaration {
public:
  /**
   * Constructor.
   */
  Model(Reference* ref);

  /**
   * Destructor.
   */
  virtual ~Model();
};
}

inline biprog::Model::Model(Reference* ref) : Declaration(ref) {
  //
}

inline biprog::Model::~Model() {
  //
}

#endif


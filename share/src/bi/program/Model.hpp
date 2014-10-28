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
class Model: public Declaration, public boost::enable_shared_from_this<Model> {
public:
  /**
   * Constructor.
   */
  Model(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Model();
};
}

inline biprog::Model::Model(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Declaration(name, brackets, parens, braces) {
  //
}

inline biprog::Model::~Model() {
  //
}

#endif


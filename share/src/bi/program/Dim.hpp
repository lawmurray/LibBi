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
class Dim: public Declaration, public boost::enable_shared_from_this<Dim> {
public:
  /**
   * Constructor.
   */
  Dim(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Dim();
};
}

inline biprog::Dim::Dim(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Declaration(name, brackets, parens, braces) {
  //
}

inline biprog::Dim::~Dim() {
  //
}

#endif

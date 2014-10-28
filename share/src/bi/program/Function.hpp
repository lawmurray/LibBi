/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_FUNCTION_HPP
#define BI_PROGRAM_FUNCTION_HPP

#include "Declaration.hpp"

namespace biprog {
/**
 * Function.
 *
 * @ingroup program
 */
class Function: public Declaration, public boost::enable_shared_from_this<
    Function> {
public:
  /**
   * Constructor.
   */
  Function(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Function();
};
}

inline biprog::Function::Function(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Declaration(name, brackets, parens, braces) {
  //
}

inline biprog::Function::~Function() {
  //
}

#endif

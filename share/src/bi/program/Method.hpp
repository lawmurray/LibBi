/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_METHOD_HPP
#define BI_PROGRAM_METHOD_HPP

#include "Declaration.hpp"

namespace biprog {
/**
 * Method.
 *
 * @ingroup program
 */
class Method: public Declaration,
    public boost::enable_shared_from_this<Method> {
public:
  /**
   * Constructor.
   */
  Method(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Method();
};
}

inline biprog::Method::Method(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Declaration(name, brackets, parens, braces) {
  //
}

inline biprog::Method::~Method() {
  //
}

#endif


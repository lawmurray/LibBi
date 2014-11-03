/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_METHOD_HPP
#define BI_PROGRAM_METHOD_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Method.
 *
 * @ingroup program
 */
class Method: public Named,
    public Parenthesised,
    public Braced,
    public boost::enable_shared_from_this<Method> {
public:
  /**
   * Constructor.
   */
  Method(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Method();
};
}

inline biprog::Method::Method(const char* name,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Named(name), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::Method::~Method() {
//
}

#endif


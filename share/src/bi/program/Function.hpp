/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_FUNCTION_HPP
#define BI_PROGRAM_FUNCTION_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Function.
 *
 * @ingroup program
 */
class Function: public Named,
    public Parenthesised,
    public Braced,
    public boost::enable_shared_from_this<Function> {
public:
  /**
   * Constructor.
   */
  Function(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Function();
};
}

inline biprog::Function::Function(const char* name,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Named(name), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::Function::~Function() {
  //
}

#endif

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
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Function.
 *
 * @ingroup program
 */
class Function: public Declaration,
    public Parenthesised,
    public Braced,
    public boost::enable_shared_from_this<Function> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param in Input statement.
   * @param out Output statement.
   * @param braces Body.
   */
  Function(const char* name, Expression* parens = NULL,
      Expression* out = NULL, Expression* braces = NULL);

  /**
   * Destructor.
   */
  virtual ~Function();

  /**
   * Output statement.
   */
  boost::shared_ptr<Expression> out;
};
}

inline biprog::Function::Function(const char* name, Expression* parens,
    Expression* out, Expression* braces) :
    Declaration(name), Parenthesised(parens), Braced(braces), out(out) {
  //
}

inline biprog::Function::~Function() {
  //
}

#endif

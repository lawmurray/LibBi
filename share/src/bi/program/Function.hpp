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
#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Bodied.hpp"

namespace biprog {
/**
 * Function.
 *
 * @ingroup program
 */
class Function: public Declaration,
    public Named,
    public Parenthesised,
    public Bodied {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param in Input statement.
   * @param out Output statement.
   * @param body Body.
   */
  Function(const char* name, Expression* in = NULL, Expression* out = NULL,
      Expression* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Function();

  /**
   * Output statement.
   */
  boost::scoped_ptr<Expression> out;
};
}

inline biprog::Function::Function(const char* name, Expression* in,
    Expression* out, Expression* body) :
    Named(name), Parenthesised(in), Bodied(body), out(out) {
  //
}

inline biprog::Function::~Function() {
  //
}

#endif

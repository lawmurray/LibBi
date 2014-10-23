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
class Function: public Declaration {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param in Input statement.
   * @param out Output statement.
   * @param body Body.
   *
   */
  Function(const char* name, Statement* in = NULL, Statement* out = NULL,
      Statement* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Function();

  /**
   * Output statement.
   */
  boost::shared_ptr<Statement> out;
};
}

inline biprog::Function::Function(const char* name, Statement* in,
    Statement* out, Statement* body) :
    Declaration(new Reference(name, in, NULL, body)), out(out) {
  //
}

inline biprog::Function::~Function() {
  //
}

#endif

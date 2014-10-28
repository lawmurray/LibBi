/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_VAR_HPP
#define BI_PROGRAM_VAR_HPP

#include "Declaration.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public Declaration,
    public Typed,
    public boost::enable_shared_from_this<Var> {
public:
  /**
   * Constructor.
   */
  Var(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces, Type* type);

  /**
   * Destructor.
   */
  virtual ~Var();
};
}

inline biprog::Var::Var(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces, Type* type) :
    Declaration(name, brackets, parens, braces), Typed(type) {
  //
}

inline biprog::Var::~Var() {
  //
}

#endif

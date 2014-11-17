/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_FUNCTIONOVERLOAD_HPP
#define BI_PROGRAM_FUNCTIONOVERLOAD_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Function.
 *
 * @ingroup program
 */
class Function: public virtual Named,
    public virtual Parenthesised,
    public virtual Braced,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Function> {
public:
  /**
   * Constructor.
   */
  Function(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Function();

  virtual boost::shared_ptr<Expression> accept(Visitor& v);

  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Function::Function(const char* name,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope) :
    Named(name), Parenthesised(parens), Braced(braces), Scoped(scope) {
  //
}

inline biprog::Function::~Function() {
  //
}

#endif

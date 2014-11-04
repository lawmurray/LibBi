/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PROGRAM_HPP
#define BI_PROGRAM_PROGRAM_HPP

#include "Scoped.hpp"
#include "Reference.hpp"

#include <deque>

namespace biprog {
/**
 * Program.
 *
 * @ingroup program
 */
class Program : public virtual Scoped {
public:
  /**
   * Constructor.
   */
  Program();

  /**
   * Destructor.
   */
  virtual ~Program();

  /**
   * Top scope on stack.
   */
  Scoped* top();

  /**
   * Push new scope on stack.
   */
  void push(Scoped* scope);

  /**
   * Pop scope from stack.
   */
  void pop();

  /**
   * Add method overload.
   */
  MethodOverload* add(MethodOverload* method);

  /**
   * Add function overload.
   */
  FunctionOverload* add(FunctionOverload* func);

  /**
   * Add any other declaration.
   */
  Named* add(Named* decl);

  /**
   * Lookup a reference, if possible.
   */
  Reference* lookup(const char* name,
      boost::shared_ptr<biprog::Expression> brackets,
      boost::shared_ptr<biprog::Expression> parens,
      boost::shared_ptr<biprog::Expression> braces);

private:
  /**
   * Scope stack.
   */
  std::deque<Scoped*> scopes;
};
}

#endif

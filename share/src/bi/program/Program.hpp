/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PROGRAM_HPP
#define BI_PROGRAM_PROGRAM_HPP

#include "Scope.hpp"
#include "Reference.hpp"

#include "boost/shared_ptr.hpp"

#include <deque>

namespace biprog {
/**
 * Program.
 *
 * @ingroup program
 */
class Program {
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
  boost::shared_ptr<Scope> top();

  /**
   * Push new scope on stack.
   */
  void push();

  /**
   * Pop scope from stack.
   */
  void pop();

  /**
   * Add a declaration.
   */
  void add(boost::shared_ptr<Expression> decl);

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
  std::deque<boost::shared_ptr<Scope> > scopes;
};
}

#endif

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
   * Get the root statement.
   */
  boost::shared_ptr<Expression> getRoot();

  /**
   * Set the root statement.
   */
  void setRoot(boost::shared_ptr<Expression> root);

  /**
   * Add a declaration.
   */
  void add(boost::shared_ptr<Expression> decl);

  /**
   * Lookup a reference. Returns an EmptyExpression if none found.
   */
  boost::shared_ptr<Expression> lookup(const char* name);

private:
  /**
   * Expression.
   */
  boost::shared_ptr<Expression> root;

  /**
   * Scope stack.
   */
  std::deque<boost::shared_ptr<Scope> > scopes;
};
}

#endif

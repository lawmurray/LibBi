/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PROGRAM_HPP
#define BI_PROGRAM_PROGRAM_HPP

#include "Typed.hpp"
#include "Scope.hpp"

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
  boost::shared_ptr<Typed> getRoot();

  /**
   * Set the root statement.
   */
  void setRoot(boost::shared_ptr<Typed> root);

  /**
   * Add a declaration.
   */
  void add(boost::shared_ptr<Typed> decl);

  /**
   * Lookup a reference. Returns an EmptyExpression if none found.
   */
  boost::shared_ptr<Typed> lookup(const char* name);

private:
  /**
   * Expression.
   */
  boost::shared_ptr<Typed> root;

  /**
   * Scope stack.
   */
  std::deque<boost::shared_ptr<Scope> > scopes;
};
}

#endif

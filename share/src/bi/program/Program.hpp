/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PROGRAM_HPP
#define BI_PROGRAM_PROGRAM_HPP

#include "Reference.hpp"
#include "../primitive/poset.hpp"
#include "../primitive/pointer_less.hpp"

#include <deque>
#include <map> /// @todo Use unordered_map types after transition to C++11

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
   * Push new scope on stack.
   */
  void push();

  /**
   * Pop scope from stack.
   */
  void pop();

  /**
   * Add any other declaration to the global scope.
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
   * @todo Should use boost::shared_ptr, but needs to be in union for Bison.
   */
  typedef Expression* pointer_type;
  typedef bi::pointer_less<Expression*> compare_type;
  typedef std::map<std::string,bi::poset<pointer_type,compare_type> > scope_type;

  /**
   * Stack scopes.
   */
  std::deque<scope_type> scopes;
};
}

#endif


/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PROGRAM_HPP
#define BI_PROGRAM_PROGRAM_HPP

#include "Declaration.hpp"
#include "Reference.hpp"

#include <deque>
#include <map> /// @todo Use unordered_map types after transition to C++11
#include <algorithm>

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
  Declaration* add(Declaration* decl);

  /**
   * Lookup a reference, if possible.
   */
  Reference* lookup(const char* name,
      boost::shared_ptr<biprog::Expression> brackets,
      boost::shared_ptr<biprog::Expression> parens,
      boost::shared_ptr<biprog::Expression> braces);

private:
  typedef std::map<std::string,biprog::Declaration*> scope_type;

  /**
   * Stack scopes.
   */
  std::deque<scope_type> scopes;
};
}

#endif


/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SCOPE_HPP
#define BI_PROGRAM_SCOPE_HPP

#include "Expression.hpp"

#include <map> ///@todo Use unordered_map after transition to C++11

#include "boost/shared_ptr.hpp"

namespace biprog {
class MethodOverload;
class FunctionOverload;
class Named;
/**
 * Scope statement.
 */
class Scope {
public:
  /**
   * Destructor.
   */
  virtual ~Scope();

  /**
   * Find declaration by name. Returns an EmptyExpression if not found.
   */
  boost::shared_ptr<Expression> find(const char* name);

  /**
   * Insert method declaration into this scope.
   */
  void add(boost::shared_ptr<MethodOverload> overload);

  /**
   * Insert function declaration into this scope.
   */
  void add(boost::shared_ptr<FunctionOverload> overload);

  /**
   * Insert any other declaration into this scope.
   */
  void add(boost::shared_ptr<Named> decl);

private:
  /**
   * Declarations within this scope.
   */
  std::map<std::string,boost::shared_ptr<Expression> > decls;
};
}

inline biprog::Scope::~Scope() {
  //
}

#endif

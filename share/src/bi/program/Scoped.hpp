/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SCOPED_HPP
#define BI_PROGRAM_SCOPED_HPP

#include "Expression.hpp"
#include "Named.hpp"

#include <map> ///@todo Use unordered_map after transition to C++11

namespace biprog {
class MethodOverload;
class FunctionOverload;
/**
 * Scoped statement.
 */
class Scoped: public virtual Expression {
public:
  /**
   * Destructor.
   */
  virtual ~Scoped() = 0;

  /**
   * Insert method declaration into this scope.
   */
  void add(MethodOverload* overload);

  /**
   * Insert function declaration into this scope.
   */
  void add(FunctionOverload* overload);

  /**
   * Insert any other declaration into this scope.
   */
  void add(Named* decl);

private:
  /**
   * Declarations within this scope.
   */
  std::map<std::string,Expression*> decls;
};
}

inline biprog::Scoped::~Scoped() {
  //
}

#endif

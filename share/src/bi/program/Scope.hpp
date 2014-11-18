/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SCOPE_HPP
#define BI_PROGRAM_SCOPE_HPP

#include "Expression.hpp"
#include "../primitive/poset.hpp"
#include "../primitive/pointer_less_or_equal.hpp"

#include <map> ///@todo Use unordered_map after transition to C++11

namespace biprog {
class Method;
class Function;
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
  //boost::shared_ptr<Expression> find(const char* name);

  /**
   * Insert any other declaration into this scope.
   */
  void add(boost::shared_ptr<Named> decl);

protected:
  typedef Expression value_type;
  typedef boost::shared_ptr<value_type> pointer_type;
  typedef bi::poset<pointer_type,bi::pointer_less_or_equal<pointer_type> > poset_type;
  typedef std::map<std::string,boost::shared_ptr<poset_type> > map_type;

  /**
   * Declarations within this scope.
   */
  map_type decls;
};
}

inline biprog::Scope::~Scope() {
  //
}

#endif

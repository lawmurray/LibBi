/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SCOPED_HPP
#define BI_PROGRAM_SCOPED_HPP

#include "Statement.hpp"
#include "Reference.hpp"
#include "Named.hpp"
#include "../exception/AmbiguousReferenceException.hpp"
#include "../primitive/poset.hpp"
#include "../primitive/pointer_less_equal.hpp"

#include <map> ///@todo Use unordered_map after transition to C++11

namespace biprog {
/**
 * Scoped statement.
 */
class Scoped {
public:
  /**
   * Constructor.
   */
  Scoped(Statement* stmt = new EmptyStatement());

  /**
   * Destructor.
   */
  virtual ~Scoped() = 0;

  /**
   * Resolve reference.
   */
  bool resolve(Reference* ref) throw (AmbiguousReferenceException);

  /**
   * Insert any other declaration into this scope.
   */
  void add(Named* decl);

  /**
   * Root statement.
   */
  Statement* stmt;

protected:
  typedef Statement value_type;
  typedef value_type* pointer_type;
  typedef bi::poset<pointer_type,bi::pointer_less_equal<pointer_type> > poset_type;
  typedef std::map<std::string,poset_type> map_type;

  /**
   * Declarations within this scope.
   */
  map_type decls;
};
}

inline biprog::Scoped::Scoped(Statement* stmt) : stmt(stmt) {
  //
}

inline biprog::Scoped::~Scoped() {
  delete stmt;
}

#endif

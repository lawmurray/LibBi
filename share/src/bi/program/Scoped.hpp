/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SCOPED_HPP
#define BI_PROGRAM_SCOPED_HPP

#include "Expression.hpp"
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
class Scoped: public virtual Expression {
public:
  /**
   * Destructor.
   */
  virtual ~Scoped() = 0;

  /**
   * Resolve reference.
   */
  bool resolve(boost::shared_ptr<Reference> ref)
      throw (AmbiguousReferenceException);

  /**
   * Insert any other declaration into this scope.
   */
  void add(boost::shared_ptr<Named> decl);

  /**
   * Root expression.
   */
  boost::shared_ptr<Typed> expr;

protected:
  typedef Named value_type;
  typedef boost::shared_ptr<value_type> pointer_type;
  typedef bi::poset<pointer_type,bi::pointer_less_equal<pointer_type> > poset_type;
  typedef std::map<std::string,boost::shared_ptr<poset_type> > map_type;

  /**
   * Declarations within this scope.
   */
  map_type decls;
};
}

inline biprog::Scoped::~Scoped() {
  //
}

#endif

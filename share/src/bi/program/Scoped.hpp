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
#include "Scope.hpp"

namespace biprog {
/**
 * Scoped statement.
 */
class Scoped: public virtual Expression {
public:
  /**
   * Constructor.
   */
  Scoped(boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Scoped() = 0;

protected:
  /**
   * Scope.
   */
  boost::shared_ptr<Scope> scope;
};
}

inline biprog::Scoped::Scoped(boost::shared_ptr<Scope> scope) :
    scope(scope) {
  /* pre-condition */
  BI_ASSERT(scope);
}

inline biprog::Scoped::~Scoped() {
  //
}

#endif

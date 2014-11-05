/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_OVERLOADED_HPP
#define BI_PROGRAM_OVERLOADED_HPP

#include "Expression.hpp"
#include "../primitive/poset.hpp"
#include "../primitive/pointer_less.hpp"

namespace biprog {
/**
 * Overloaded declaration.
 */
class Overloaded: public virtual Expression {
public:
  /**
   * Destructor.
   */
  virtual ~Overloaded() = 0;

  /**
   * Add overload.
   */
  void add(boost::shared_ptr<Expression> overload);

protected:
  typedef boost::shared_ptr<Expression> pointer_type;

  /**
   * Overloads.
   */
  bi::poset<pointer_type,bi::pointer_less<pointer_type> > overloads;
};
}

inline biprog::Overloaded::~Overloaded() {
  //
}

inline void biprog::Overloaded::add(boost::shared_ptr<Expression> overload) {
  overloads.insert(overload);
}

#endif

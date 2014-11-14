/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_OVERLOADED_HPP
#define BI_PROGRAM_OVERLOADED_HPP

#include "Expression.hpp"
#include "../primitive/poset.hpp"
#include "../primitive/pointer_less.hpp"

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include "boost/enable_shared_from_this.hpp"

namespace biprog {
/**
 * Overloaded declaration.
 *
 * @ingroup program
 */
class Overloaded {
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

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
class Overloaded : public virtual Expression {
public:
  /**
   * Destructor.
   */
  virtual ~Overloaded() = 0;

  /**
   * Add overload.
   */
  void add(Expression* overload);

private:
  /**
   * Overloads.
   */
  bi::poset<Expression*,bi::pointer_less<Expression*> > overloads;
};
}

inline biprog::Overloaded::~Overloaded() {
  //
}

inline void biprog::Overloaded::add(Expression* overload) {
  overloads.insert(overload);
}

#endif

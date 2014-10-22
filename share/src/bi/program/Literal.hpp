/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LITERAL_HPP
#define BI_PROGRAM_LITERAL_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Literal.
 *
 * @ingroup program
 *
 * @todo Flyweight this.
 */
template<class T1>
class Literal: public Expression {
public:
  /**
   * Constructor.
   */
  Literal(const T1& value);

  /**
   * Destructor.
   */
  virtual ~Literal();

  /**
   * Value.
   */
  T1 value;
};
}

template<class T1>
inline biprog::Literal<T1>::Literal(const T1& value) : value(value) {
  //
}

template<class T1>
inline biprog::Literal<T1>::~Literal() {
  //
}

#endif

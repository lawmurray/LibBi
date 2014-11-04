/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
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
class Literal: public virtual Expression,
    public boost::enable_shared_from_this<Literal<T1> > {
public:
  /**
   * Constructor.
   */
  Literal(const T1& value);

  /**
   * Destructor.
   */
  virtual ~Literal();

  /*
   * Operators.
   */
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;

  /**
   * Value.
   */
  T1 value;
};
}

template<class T1>
inline biprog::Literal<T1>::Literal(const T1& value) :
    value(value) {
  //
}

template<class T1>
inline biprog::Literal<T1>::~Literal() {
  //
}

template<class T1>
inline bool biprog::Literal<T1>::operator<(const Expression& o) const {
  return false;
}

template<class T1>
inline bool biprog::Literal<T1>::operator<=(const Expression& o) const {
  return operator==(o);
}

template<class T1>
inline bool biprog::Literal<T1>::operator>(const Expression& o) const {
  return false;
}

template<class T1>
inline bool biprog::Literal<T1>::operator>=(const Expression& o) const {
  return operator==(o);
}

template<class T1>
inline bool biprog::Literal<T1>::operator==(const Expression& o) const {
  try {
    const Literal<T1>& expr = dynamic_cast<const Literal<T1>&>(o);
    return value == expr.value;
  } catch (std::bad_cast e) {
    return false;
  }
}

template<class T1>
inline bool biprog::Literal<T1>::operator!=(const Expression& o) const {
  try {
    const Literal<T1>& expr = dynamic_cast<const Literal<T1>&>(o);
    return value != expr.value;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif

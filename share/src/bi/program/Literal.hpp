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
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const Literal& o) const;
  virtual bool operator<=(const Literal& o) const;
  virtual bool operator>(const Literal& o) const;
  virtual bool operator>=(const Literal& o) const;
  virtual bool operator==(const Literal& o) const;
  virtual bool operator!=(const Literal& o) const;

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
inline bool biprog::Literal<T1>::operator<(const Literal<T1>& o) const {
  return false;
}

template<class T1>
inline bool biprog::Literal<T1>::operator<=(const Literal<T1>& o) const {
  return operator==(o);
}

template<class T1>
inline bool biprog::Literal<T1>::operator>(const Literal<T1>& o) const {
  return false;
}

template<class T1>
inline bool biprog::Literal<T1>::operator>=(const Literal<T1>& o) const {
  return operator==(o);
}

template<class T1>
inline bool biprog::Literal<T1>::operator==(const Literal<T1>& o) const {
  return value == o.value;
}

template<class T1>
inline bool biprog::Literal<T1>::operator!=(const Literal<T1>& o) const {
  return value != o.value;
}

#endif

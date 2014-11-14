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
    public virtual boost::enable_shared_from_this<Literal<T1> > {
public:
  /**
   * Constructor.
   */
  Literal(const T1& value);

  /**
   * Destructor.
   */
  virtual ~Literal();

  virtual boost::shared_ptr<Expression> accept(Visitor& v);

  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;

  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

protected:
  /**
   * Value.
   */
  T1 value;
};
}

#include "../visitor/Visitor.hpp"

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
boost::shared_ptr<biprog::Expression> biprog::Literal<T1>::accept(
    Visitor& v) {
  return v.visit(this->shared_from_this());
}

template<class T1>
bool biprog::Literal<T1>::operator<(const Expression& o) const {
  return false;
}

template<class T1>
bool biprog::Literal<T1>::operator<=(const Expression& o) const {
  return operator==(o);
}

template<class T1>
bool biprog::Literal<T1>::operator>(const Expression& o) const {
  return false;
}

template<class T1>
bool biprog::Literal<T1>::operator>=(const Expression& o) const {
  return operator==(o);
}

template<class T1>
bool biprog::Literal<T1>::operator==(const Expression& o) const {
  try {
    const Literal<T1>& expr = dynamic_cast<const Literal<T1>&>(o);
    return value == expr.value;
  } catch (std::bad_cast e) {
    return false;
  }
}

template<class T1>
bool biprog::Literal<T1>::operator!=(const Expression& o) const {
  try {
    const Literal<T1>& expr = dynamic_cast<const Literal<T1>&>(o);
    return value != expr.value;
  } catch (std::bad_cast e) {
    return true;
  }
}

template<class T1>
void biprog::Literal<T1>::output(std::ostream& out) const {
  out << value;
}

#endif

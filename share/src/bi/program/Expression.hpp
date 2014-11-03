/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "boost/shared_ptr.hpp"
#include "boost/enable_shared_from_this.hpp"

#define BI_EXPRESSION_OP(op, type) \
  virtual bool operator op(const type& o) const { \
    return false; \
  }

namespace biprog {
/*
 * Forward declarations of concrete types.
 */
class BinaryExpression;
class Sequence;
class UnaryExpression;
class Loop;
class Conditional;
class Function;
class Method;
class Model;
class Reference;
class Type;
class Dim;
class Var;

/**
 * Expression.
 *
 * @ingroup program
 */
class Expression: public boost::enable_shared_from_this<Expression> {
public:
  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /**
   * @name Abstract operators
   *
   * These operators compare expressions based on specialisation. They are
   * used for selecting the most specialised function/method overload for
   * any reference. A double virtual function call is used to ensure that
   * the comparison of any two objects is of their most-derived type, i.e.
   *
   *  @code
   *  Expression* x = new BinaryExpression(...);
   *  Expression* y = new UnaryExpression(...);
   *  @endcode
   *
   * Now, <tt>*x == *y</tt> resolves to
   * <tt>Expression::operator==(UnaryExpression&)</tt>, which internally
   * calls <tt>UnaryExpression::operator==(BinaryExpression&)</tt>; now the
   * comparison is made using the most-derived types.
   */
  //@{
  virtual bool operator<(const Expression& o) const {
    return o >= *this;
  }

  virtual bool operator<=(const Expression& o) const {
    return o > *this;
  }

  virtual bool operator>(const Expression& o) const {
    return o <= *this;
  }

  virtual bool operator>=(const Expression& o) const {
    return o < *this;
  }

  virtual bool operator==(const Expression& o) const {
    return o == *this;
  }

  virtual bool operator!=(const Expression& o) const {
    return o != *this;
  }
  //@}

  /**
   * @name Concrete operators
   *
   * These operators work on derived types. They return false for
   * everything. at this level. Derived types should override those that may
   * return true (i.e. any where the argument is the same as that derived
   * type).
   */
  //@{
  BI_EXPRESSION_OP(<, BinaryExpression)
  BI_EXPRESSION_OP(<=, BinaryExpression)
  BI_EXPRESSION_OP(>, BinaryExpression)
  BI_EXPRESSION_OP(>=, BinaryExpression)
  BI_EXPRESSION_OP(==, BinaryExpression)
  BI_EXPRESSION_OP(!=, BinaryExpression)

  BI_EXPRESSION_OP(<, Sequence)
  BI_EXPRESSION_OP(<=, Sequence)
  BI_EXPRESSION_OP(>, Sequence)
  BI_EXPRESSION_OP(>=, Sequence)
  BI_EXPRESSION_OP(==, Sequence)
  BI_EXPRESSION_OP(!=, Sequence)

  BI_EXPRESSION_OP(<, UnaryExpression)
  BI_EXPRESSION_OP(<=, UnaryExpression)
  BI_EXPRESSION_OP(>, UnaryExpression)
  BI_EXPRESSION_OP(>=, UnaryExpression)
  BI_EXPRESSION_OP(==, UnaryExpression)
  BI_EXPRESSION_OP(!=, UnaryExpression)

  BI_EXPRESSION_OP(<, Loop)
  BI_EXPRESSION_OP(<=, Loop)
  BI_EXPRESSION_OP(>, Loop)
  BI_EXPRESSION_OP(>=, Loop)
  BI_EXPRESSION_OP(==, Loop)
  BI_EXPRESSION_OP(!=, Loop)

  BI_EXPRESSION_OP(<, Conditional)
  BI_EXPRESSION_OP(<=, Conditional)
  BI_EXPRESSION_OP(>, Conditional)
  BI_EXPRESSION_OP(>=, Conditional)
  BI_EXPRESSION_OP(==, Conditional)
  BI_EXPRESSION_OP(!=, Conditional)

  BI_EXPRESSION_OP(<, Function)
  BI_EXPRESSION_OP(<=, Function)
  BI_EXPRESSION_OP(>, Function)
  BI_EXPRESSION_OP(>=, Function)
  BI_EXPRESSION_OP(==, Function)
  BI_EXPRESSION_OP(!=, Function)

  BI_EXPRESSION_OP(<, Method)
  BI_EXPRESSION_OP(<=, Method)
  BI_EXPRESSION_OP(>, Method)
  BI_EXPRESSION_OP(>=, Method)
  BI_EXPRESSION_OP(==, Method)
  BI_EXPRESSION_OP(!=, Method)

  BI_EXPRESSION_OP(<, Model)
  BI_EXPRESSION_OP(<=, Model)
  BI_EXPRESSION_OP(>, Model)
  BI_EXPRESSION_OP(>=, Model)
  BI_EXPRESSION_OP(==, Model)
  BI_EXPRESSION_OP(!=, Model)

  BI_EXPRESSION_OP(<, Reference)
  BI_EXPRESSION_OP(<=, Reference)
  BI_EXPRESSION_OP(>, Reference)
  BI_EXPRESSION_OP(>=, Reference)
  BI_EXPRESSION_OP(==, Reference)
  BI_EXPRESSION_OP(!=, Reference)

  BI_EXPRESSION_OP(<, Type)
  BI_EXPRESSION_OP(<=, Type)
  BI_EXPRESSION_OP(>, Type)
  BI_EXPRESSION_OP(>=, Type)
  BI_EXPRESSION_OP(==, Type)
  BI_EXPRESSION_OP(!=, Type)

  BI_EXPRESSION_OP(<, Dim)
  BI_EXPRESSION_OP(<=, Dim)
  BI_EXPRESSION_OP(>, Dim)
  BI_EXPRESSION_OP(>=, Dim)
  BI_EXPRESSION_OP(==, Dim)
  BI_EXPRESSION_OP(!=, Dim)

  BI_EXPRESSION_OP(<, Var)
  BI_EXPRESSION_OP(<=, Var)
  BI_EXPRESSION_OP(>, Var)
  BI_EXPRESSION_OP(>=, Var)
  BI_EXPRESSION_OP(==, Var)
  BI_EXPRESSION_OP(!=, Var)
  //@}
};
}

inline biprog::Expression::~Expression() {
  //
}

#endif


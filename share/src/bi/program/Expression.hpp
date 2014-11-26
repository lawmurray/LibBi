/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "Statement.hpp"
#include "../misc/assert.hpp"

namespace biprog {
class Visitor;

/**
 * Expression object.
 *
 * @ingroup program
 */
class Expression {
public:
  /**
   * Constructor.
   *
   * @param type type.
   */
  Expression();

  /**
   * Constructor.
   *
   * @param type type.
   */
  Expression(Statement* type);

  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /**
   * Clone expression.
   */
  virtual Expression* clone() = 0;

  /**
   * Accept visitor.
   *
   * @param v The visitor.
   *
   * @return New expression with which to replace this one (may be the same).
   */
  virtual Expression* acceptExpression(Visitor& v) = 0;

  /*
   * Bool cast to check for non-empty expression.
   */
  virtual operator bool() const;

  /*
   * Comparison operators for comparing expressions in terms of
   * specialisation.
   *
   * The first two are the most commonly used, and so overridden by derived
   * classes. The remainder are expressed in terms of these.
   */
  virtual bool operator<=(const Expression& o) const = 0;
  virtual bool operator==(const Expression& o) const = 0;
  bool operator<(const Expression& o) const;
  bool operator>(const Expression& o) const;
  bool operator>=(const Expression& o) const;
  bool operator!=(const Expression& o) const;

  /**
   * Type.
   */
  Statement* type;
};
}

#include "EmptyStatement.hpp"

inline biprog::Expression::Expression() :
    type(new EmptyStatement()) {
  //
}

inline biprog::Expression::Expression(Statement* type) :
    type(type) {
  //
}

inline biprog::Expression::~Expression() {
  delete type;
}

inline biprog::Expression::operator bool() const {
  return true;
}

inline bool biprog::Expression::operator<(const Expression& o) const {
  return *this <= o && *this != o;
}

inline bool biprog::Expression::operator>(const Expression& o) const {
  return !(*this <= o);
}

inline bool biprog::Expression::operator>=(const Expression& o) const {
  return !(*this < o);
}

inline bool biprog::Expression::operator!=(const Expression& o) const {
  return !(*this == o);
}

#endif

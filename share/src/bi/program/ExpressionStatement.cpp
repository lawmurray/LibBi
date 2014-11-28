/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ExpressionStatement.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::ExpressionStatement* biprog::ExpressionStatement::clone() {
  return new ExpressionStatement(expr->clone());
}

biprog::Statement* biprog::ExpressionStatement::acceptStatement(Visitor& v) {
  expr = expr->acceptExpression(v);

  return v.visitStatement(this);
}

bool biprog::ExpressionStatement::operator<=(const Statement& o) const {
  try {
    const ExpressionStatement& o1 = dynamic_cast<const ExpressionStatement&>(o);
    return *expr <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return !*o1.brackets && !*o1.parens && !*o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::ExpressionStatement::operator==(const Statement& o) const {
  try {
    const ExpressionStatement& o1 = dynamic_cast<const ExpressionStatement&>(o);
    return *expr == *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::ExpressionStatement::output(std::ostream& out) const {
  out << *expr << ';';
}

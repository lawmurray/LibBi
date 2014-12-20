/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ReturnStatement.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::ReturnStatement* biprog::ReturnStatement::clone() {
  return new ReturnStatement(expr->clone());
}

biprog::Statement* biprog::ReturnStatement::acceptStatement(Visitor& v) {
  expr = expr->acceptExpression(v);

  return v.visitStatement(this);
}

bool biprog::ReturnStatement::operator<=(const Statement& o) const {
  try {
    const ReturnStatement& o1 = dynamic_cast<const ReturnStatement&>(o);
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

bool biprog::ReturnStatement::operator==(const Statement& o) const {
  try {
    const ReturnStatement& o1 = dynamic_cast<const ReturnStatement&>(o);
    return *expr == *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::ReturnStatement::output(std::ostream& out) const {
  out << "return " << *expr << ';';
}

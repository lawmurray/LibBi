/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ReturnExpression.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Typed* biprog::ReturnExpression::clone() {
  return new ReturnExpression(expr->clone());
}

biprog::Typed* biprog::ReturnExpression::accept(Visitor& v) {
  type = type->accept(v);
  expr = expr->accept(v);
  return v.visit(this);
}

bool biprog::ReturnExpression::operator<=(const Typed& o) const {
  try {
    const ReturnExpression& o1 = dynamic_cast<const ReturnExpression&>(o);
    return *expr <= *o1.expr && *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return !*o1.brackets && !*o1.parens && !*o1.braces && *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::ReturnExpression::operator==(const Typed& o) const {
  try {
    const ReturnExpression& o1 = dynamic_cast<const ReturnExpression&>(o);
    return *expr == *o1.expr && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::ReturnExpression::output(std::ostream& out) const {
  out << "return " << *expr;
}

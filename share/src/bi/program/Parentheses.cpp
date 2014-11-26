/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirgroup.au>
 * $Rev$
 * $Date$
 */
#include "Parentheses.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Parentheses* biprog::Parentheses::clone() {
  return new Parentheses(expr->clone());
}

biprog::Expression* biprog::Parentheses::acceptExpression(Visitor& v) {
  type = type->acceptStatement(v);
  expr = expr->acceptExpression(v);

  return v.visitExpression(this);
}

bool biprog::Parentheses::operator<=(const Expression& o) const {
  try {
    const Parentheses& o1 = dynamic_cast<const Parentheses&>(o);
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

bool biprog::Parentheses::operator==(const Expression& o) const {
  try {
    const Parentheses& o1 = dynamic_cast<const Parentheses&>(o);
    return *expr == *o1.expr && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Parentheses::output(std::ostream& out) const {
  out << '(' << *expr << ')';
}

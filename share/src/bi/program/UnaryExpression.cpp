/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "UnaryExpression.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::UnaryExpression* biprog::UnaryExpression::clone() {
  return new UnaryExpression(op, right->clone());
}

biprog::Expression* biprog::UnaryExpression::acceptExpression(Visitor& v) {
  type = type->acceptStatement(v);
  right = right->acceptExpression(v);

  return v.visitExpression(this);
}

bool biprog::UnaryExpression::operator<=(const Expression& o) const {
  try {
    const UnaryExpression& o1 = dynamic_cast<const UnaryExpression&>(o);
    return op == o1.op && *right <= *o1.right && *type <= *o1.type;
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

bool biprog::UnaryExpression::operator==(const Expression& o) const {
  try {
    const UnaryExpression& o1 = dynamic_cast<const UnaryExpression&>(o);
    return op == o1.op && *right == *o1.right && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::UnaryExpression::output(std::ostream& out) const {
  out << op << *right;
}

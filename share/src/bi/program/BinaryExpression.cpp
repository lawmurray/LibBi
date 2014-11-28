/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "BinaryExpression.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::BinaryExpression* biprog::BinaryExpression::clone() {
  return new BinaryExpression(left->clone(), op, right->clone());
}

biprog::Expression* biprog::BinaryExpression::acceptExpression(Visitor& v) {
  type = type->acceptStatement(v);
  left = left->acceptExpression(v);
  right = right->acceptExpression(v);

  return v.visitExpression(this);
}

bool biprog::BinaryExpression::operator<=(const Expression& o) const {
  try {
    const BinaryExpression& o1 = dynamic_cast<const BinaryExpression&>(o);
    return op == o1.op && *left <= *o1.left && *right <= *o1.right
        && *type <= *o1.type;
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

bool biprog::BinaryExpression::operator==(const Expression& o) const {
  try {
    const BinaryExpression& o1 = dynamic_cast<const BinaryExpression&>(o);
    return op == o1.op && *left == *o1.left && *right == *o1.right
        && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::BinaryExpression::output(std::ostream& out) const {
  out << *left << ops[op] << *right;
}

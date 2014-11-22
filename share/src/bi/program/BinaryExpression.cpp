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

boost::shared_ptr<biprog::Typed> biprog::BinaryExpression::accept(
    Visitor& v) {
  type = type->accept(v);
  left = left->accept(v);
  right = right->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::BinaryExpression::operator<=(const Typed& o) const {
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

bool biprog::BinaryExpression::operator==(const Typed& o) const {
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
  out << *left << op << *right;
}

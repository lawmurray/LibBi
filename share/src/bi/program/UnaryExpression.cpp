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

boost::shared_ptr<biprog::Typed> biprog::UnaryExpression::accept(Visitor& v) {
  type = type->accept(v);
  right = right->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::UnaryExpression::operator<=(const Typed& o) const {
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

bool biprog::UnaryExpression::operator==(const Typed& o) const {
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

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "UnaryExpression.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::UnaryExpression::accept(
    Visitor& v) {
  right = right->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::UnaryExpression::operator<=(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right <= *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::UnaryExpression::operator==(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right == *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::UnaryExpression::output(std::ostream& out) const {
  out << op << *right;
}

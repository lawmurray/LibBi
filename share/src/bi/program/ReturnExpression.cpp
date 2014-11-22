/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "ReturnExpression.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::ReturnExpression::accept(
    Visitor& v) {
  type = type->accept(v);
  expr = expr->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::ReturnExpression::operator<=(const Typed& o) const {
  try {
    const ReturnExpression& ret = dynamic_cast<const ReturnExpression&>(o);
    return *expr <= *ret.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::ReturnExpression::operator==(const Typed& o) const {
  try {
    const ReturnExpression& ret = dynamic_cast<const ReturnExpression&>(o);
    return *expr == *ret.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::ReturnExpression::output(std::ostream& out) const {
  out << "return " << *expr;
}

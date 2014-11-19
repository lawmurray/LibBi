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

boost::shared_ptr<biprog::Expression> biprog::BinaryExpression::accept(
    Visitor& v) {
  left = left->accept(v);
  right = right->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::BinaryExpression::operator<=(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op == expr.op && *left <= *expr.left && *right <= *expr.right;
  } catch (std::bad_cast e) {
    //
  }
//  try {
//    const Reference& expr = dynamic_cast<const Reference&>(o);
//    return type <= *expr.type && !*expr.brackets && !*expr.parens && !*expr.braces;
//  } catch (std::bad_cast e) {
//    //
//  }
  return false;
}

bool biprog::BinaryExpression::operator==(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op == expr.op && *left == *expr.left && *right == *expr.right;
  } catch (std::bad_cast e) {
    //
  }
//  try {
//    const Reference& expr = dynamic_cast<const Reference&>(o);
//    return type < *expr.type && !*expr.brackets && !*expr.parens && !*expr.braces;
//  } catch (std::bad_cast e) {
//    //
//  }
  return false;
}

void biprog::BinaryExpression::output(std::ostream& out) const {
  out << *left << op << *right;
}

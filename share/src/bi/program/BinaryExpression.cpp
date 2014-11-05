/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "BinaryExpression.hpp"

bool biprog::BinaryExpression::operator<(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op == expr.op && *left < *expr.left && *right < *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::BinaryExpression::operator<=(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op == expr.op && *left <= *expr.left && *right <= *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::BinaryExpression::operator>(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op == expr.op && *left > *expr.left && *right > *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::BinaryExpression::operator>=(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op == expr.op && *left >= *expr.left && *right >= *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::BinaryExpression::operator==(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op == expr.op && *left == *expr.left && *right == *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::BinaryExpression::operator!=(const Expression& o) const {
  try {
    const BinaryExpression& expr = dynamic_cast<const BinaryExpression&>(o);
    return op != expr.op || *left != *expr.left || *right != *expr.right;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::BinaryExpression::output(std::ostream& out) const {
  out << *left << op << *right;
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "UnaryExpression.hpp"

bool biprog::UnaryExpression::operator<(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right < *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::UnaryExpression::operator<=(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right <= *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::UnaryExpression::operator>(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right > *expr.right;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::UnaryExpression::operator>=(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op == expr.op && *right >= *expr.right;
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

bool biprog::UnaryExpression::operator!=(const Expression& o) const {
  try {
    const UnaryExpression& expr = dynamic_cast<const UnaryExpression&>(o);
    return op != expr.op || *right != *expr.right;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::UnaryExpression::output(std::ostream& out) const {
  out << op << *right;
}

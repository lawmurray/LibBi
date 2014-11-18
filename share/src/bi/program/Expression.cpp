/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Expression.hpp"

biprog::Expression::operator bool() const {
  return true;
}

bool biprog::Expression::operator<(const Expression& o) const {
  return false;
}

bool biprog::Expression::operator<=(const Expression& o) const {
  return false;
}

bool biprog::Expression::operator>(const Expression& o) const {
  return false;
}

bool biprog::Expression::operator>=(const Expression& o) const {
  return false;
}

bool biprog::Expression::operator==(const Expression& o) const {
  return false;
}

bool biprog::Expression::operator!=(const Expression& o) const {
  return true;
}

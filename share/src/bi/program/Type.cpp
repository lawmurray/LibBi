/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Type.hpp"

bool biprog::Type::operator<(const Expression& o) const {
  return false;
}

bool biprog::Type::operator<=(const Expression& o) const {
  return operator==(o);
}

bool biprog::Type::operator>(const Expression& o) const {
  return false;
}

bool biprog::Type::operator>=(const Expression& o) const {
  return operator==(o);
}

bool biprog::Type::operator==(const Expression& o) const {
  try {
    const Type& expr = dynamic_cast<const Type&>(o);
    ///@todo Avoid string comparison.
    return name.compare(expr.name) == 0;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Type::operator!=(const Expression& o) const {
  try {
    const Type& expr = dynamic_cast<const Type&>(o);
    ///@todo Avoid string comparison.
    return name.compare(expr.name) != 0;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Type::output(std::ostream& out) const {
  out << name;
}

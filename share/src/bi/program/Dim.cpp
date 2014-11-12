/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Dim.hpp"

bool biprog::Dim::operator<(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets < *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Dim::operator<=(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets <= *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Dim::operator>(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets > *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Dim::operator>=(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets >= *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Dim::operator==(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets == *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Dim::operator!=(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets != *expr.brackets;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Dim::output(std::ostream& out) const {
  out << "dim " << name << *brackets;
}

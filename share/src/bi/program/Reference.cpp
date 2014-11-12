/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Reference.hpp"

bool biprog::Reference::operator<(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets < *expr.brackets && *parens < *expr.parens
        && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Reference::operator<=(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets <= *expr.brackets && *parens <= *expr.parens
        && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Reference::operator>(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets > *expr.brackets && *parens > *expr.parens
        && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Reference::operator>=(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets >= *expr.brackets && *parens >= *expr.parens
        && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Reference::operator==(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets == *expr.brackets && *parens == *expr.parens
        && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Reference::operator!=(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets != *expr.brackets || *parens != *expr.parens
        || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Reference::output(std::ostream& out) const {
  out << name << *brackets << *parens << *braces;
}

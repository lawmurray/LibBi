/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Model.hpp"

bool biprog::Model::operator<(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens < *expr.parens && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Model::operator<=(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Model::operator>(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens > *expr.parens && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Model::operator>=(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens >= *expr.parens && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Model::operator==(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Model::operator!=(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens != *expr.parens || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Model::output(std::ostream& out) const {
  out << "model " << name << *parens << ' ' << *braces;
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Var.hpp"

bool biprog::Var::operator<(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets < *expr.brackets && *type < *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Var::operator<=(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets <= *expr.brackets && *type <= *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Var::operator>(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets > *expr.brackets && *type > *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Var::operator>=(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets >= *expr.brackets && *type >= *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Var::operator==(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets == *expr.brackets && *type == *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Var::operator!=(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets != *expr.brackets || *type != *expr.type;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Var::output(std::ostream& out) const {
  out << "var " << name << *brackets << ':' << *type;
}

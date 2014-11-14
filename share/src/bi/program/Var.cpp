/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Var.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Var::accept(Visitor& v) {
  brackets = brackets->accept(v);
  type = type->accept(v);
  return v.visit(shared_from_this());
}

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

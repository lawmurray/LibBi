/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Class.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Class::accept(Visitor& v) {
  parens = parens->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Class::operator<(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens < *expr.parens && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Class::operator<=(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Class::operator>(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens > *expr.parens && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Class::operator>=(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens >= *expr.parens && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Class::operator==(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Class::operator!=(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens != *expr.parens || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Class::output(std::ostream& out) const {
  out << "class " << name << *parens << ' ' << *braces;
}

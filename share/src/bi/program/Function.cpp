/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Function.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Function::accept(
    Visitor& v) {
  parens = parens->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Function::operator<(const Expression& o) const {
  try {
    const Function& expr = dynamic_cast<const Function&>(o);
    return *parens < *expr.parens && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Function::operator<=(const Expression& o) const {
  try {
    const Function& expr = dynamic_cast<const Function&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Function::operator>(const Expression& o) const {
  try {
    const Function& expr = dynamic_cast<const Function&>(o);
    return *parens > *expr.parens && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Function::operator>=(const Expression& o) const {
  try {
    const Function& expr = dynamic_cast<const Function&>(o);
    return *parens >= *expr.parens && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Function::operator==(const Expression& o) const {
  try {
    const Function& expr = dynamic_cast<const Function&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Function::operator!=(const Expression& o) const {
  try {
    const Function& expr = dynamic_cast<const Function&>(o);
    return *parens != *expr.parens || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Function::output(std::ostream& out) const {
  out << "function " << name << *parens << ' ' << *braces;
}

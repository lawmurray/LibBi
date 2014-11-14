/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "FunctionOverload.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::FunctionOverload::accept(
    Visitor& v) {
  parens = parens->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::FunctionOverload::operator<(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens < *expr.parens && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::FunctionOverload::operator<=(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::FunctionOverload::operator>(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens > *expr.parens && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::FunctionOverload::operator>=(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens >= *expr.parens && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::FunctionOverload::operator==(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::FunctionOverload::operator!=(const Expression& o) const {
  try {
    const FunctionOverload& expr = dynamic_cast<const FunctionOverload&>(o);
    return *parens != *expr.parens || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::FunctionOverload::output(std::ostream& out) const {
  out << "function " << name << *parens << ' ' << *braces;
}

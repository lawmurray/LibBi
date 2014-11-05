/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "FunctionOverload.hpp"

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
  out << "function " << name << *parens << *braces;
}

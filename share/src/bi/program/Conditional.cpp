/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Conditional.hpp"

bool biprog::Conditional::operator<(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond < *expr.cond && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Conditional::operator<=(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond <= *expr.cond && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Conditional::operator>(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond > *expr.cond && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Conditional::operator>=(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond >= *expr.cond && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Conditional::operator==(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond == *expr.cond && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Conditional::operator!=(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond != *expr.cond || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Conditional::output(std::ostream& out) const {
  out << "if " << *cond << *braces;
}

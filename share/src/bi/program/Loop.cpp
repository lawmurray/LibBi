/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Loop.hpp"

bool biprog::Loop::operator<(const Expression& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond < *expr.cond && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Loop::operator<=(const Expression& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond <= *expr.cond && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Loop::operator>(const Expression& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond > *expr.cond && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Loop::operator>=(const Expression& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond >= *expr.cond && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Loop::operator==(const Expression& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond == *expr.cond && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Loop::operator!=(const Expression& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond != *expr.cond || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Loop::output(std::ostream& out) const {
  out << "while " << *cond << *braces;
}

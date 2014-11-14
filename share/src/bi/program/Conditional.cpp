/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Conditional.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Conditional::accept(
    Visitor& v) {
  cond = cond->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

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

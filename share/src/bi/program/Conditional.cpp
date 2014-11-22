/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Conditional.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Conditional::accept(
    Visitor& v) {
  type = type->accept(v);
  cond = cond->accept(v);
  braces = braces->accept(v);
  falseBraces = falseBraces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Conditional::operator<=(const Typed& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond <= *expr.cond && *braces <= *expr.braces
        && *falseBraces <= *expr.falseBraces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Conditional::operator==(const Typed& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond == *expr.cond && *braces == *expr.braces
        && *falseBraces == *expr.falseBraces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Conditional::output(std::ostream& out) const {
  out << "if " << *cond << ' ' << *braces;
  if (*falseBraces) {
    out << " else " << *falseBraces;
  }
}

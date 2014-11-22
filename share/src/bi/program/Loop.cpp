/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Loop.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Loop::accept(Visitor& v) {
  type = type->accept(v);
  cond = cond->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Loop::operator<=(const Typed& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond <= *expr.cond && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Loop::operator==(const Typed& o) const {
  try {
    const Loop& expr = dynamic_cast<const Loop&>(o);
    return *cond == *expr.cond && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Loop::output(std::ostream& out) const {
  out << "while " << *cond << *braces;
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Loop.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Typed* biprog::Loop::clone() {
  return new Loop(cond->clone(), braces->clone());
}

biprog::Typed* biprog::Loop::accept(Visitor& v) {
  type = type->accept(v);
  cond = cond->accept(v);
  braces = braces->accept(v);
  return v.visit(this);
}

bool biprog::Loop::operator<=(const Typed& o) const {
  try {
    const Loop& o1 = dynamic_cast<const Loop&>(o);
    return *cond <= *o1.cond && *braces <= *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return !*o1.brackets && !*o1.parens && !*o1.braces && *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Loop::operator==(const Typed& o) const {
  try {
    const Loop& o1 = dynamic_cast<const Loop&>(o);
    return *cond == *o1.cond && *braces == *o1.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Loop::output(std::ostream& out) const {
  out << "while " << *cond << ' ' << *braces;
}

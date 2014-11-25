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

biprog::Loop* biprog::Loop::clone() {
  return new Loop(cond->clone(), braces->clone());
}

biprog::Statement* biprog::Loop::accept(Visitor& v) {
  cond = cond->accept(v);
  braces = braces->accept(v);
  return v.visit(this);
}

bool biprog::Loop::operator<=(const Statement& o) const {
  try {
    const Loop& o1 = dynamic_cast<const Loop&>(o);
    return *cond <= *o1.cond && *braces <= *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return !*o1.brackets && !*o1.parens && !*o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Loop::operator==(const Statement& o) const {
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

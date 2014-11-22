/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Def.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Def::accept(Visitor& v) {
  parens = parens->accept(v);
  type = type->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Def::operator<=(const Typed& o) const {
  try {
    const Def& o1 = dynamic_cast<const Def&>(o);
    return *parens <= *o1.parens && *type <= *o1.type && *braces <= *o1.braces;
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

bool biprog::Def::operator==(const Typed& o) const {
  try {
    const Def& o1 = dynamic_cast<const Def&>(o);
    return *parens == *o1.parens && *type == *o1.type && *braces == *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Def::output(std::ostream& out) const {
  out << "def " << name;
  if (*parens) {
    out << *parens;
  }
  if (*type) {
    out << ':' << *type;
  }
  if (*braces) {
    out << ' ' << *braces;
  }
}

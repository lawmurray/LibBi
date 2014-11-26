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

#include <typeinfo>

biprog::Def* biprog::Def::clone() {
  return new Def(name, parens->clone(), type->clone(), braces->clone());
}

biprog::Statement* biprog::Def::acceptStatement(Visitor& v) {
  parens = parens->acceptExpression(v);
  type = type->acceptStatement(v);
  braces = braces->acceptExpression(v);

  return v.visitStatement(this);
}

bool biprog::Def::operator<=(const Statement& o) const {
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

bool biprog::Def::operator==(const Statement& o) const {
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
  } else {
    out << ';';
  }
}

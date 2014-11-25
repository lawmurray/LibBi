/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Var.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Typed* biprog::Var::clone() {
  return new Var(name, brackets->clone(), type->clone());
}

biprog::Typed* biprog::Var::accept(Visitor& v) {
  brackets = brackets->accept(v);
  type = type->accept(v);
  return v.visit(this);
}

bool biprog::Var::operator<=(const Typed& o) const {
  try {
    const Var& o1 = dynamic_cast<const Var&>(o);
    return *brackets <= *o1.brackets && *type <= *o1.type;
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

bool biprog::Var::operator==(const Typed& o) const {
  try {
    const Var& o1 = dynamic_cast<const Var&>(o);
    return *brackets == *o1.brackets && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Var::output(std::ostream& out) const {
  out << "var " << name << *brackets << ':' << *type;
}

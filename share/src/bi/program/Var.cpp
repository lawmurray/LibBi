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

biprog::Var* biprog::Var::clone() {
  return new Var(name, brackets->clone(), type->clone());
}

biprog::Statement* biprog::Var::acceptStatement(Visitor& v) {
  type = type->acceptStatement(v);
  brackets = brackets->acceptExpression(v);

  return v.visitStatement(this);
}

bool biprog::Var::operator<=(const Statement& o) const {
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

bool biprog::Var::operator==(const Statement& o) const {
  try {
    const Var& o1 = dynamic_cast<const Var&>(o);
    return *brackets == *o1.brackets && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Var::output(std::ostream& out) const {
  out << "var " << name << *brackets << ':' << *type << ';';
}

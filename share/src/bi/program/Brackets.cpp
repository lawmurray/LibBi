/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirgroup.au>
 * $Rev$
 * $Date$
 */
#include "Brackets.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Typed* biprog::Brackets::clone() {
  return new Brackets(expr->clone());
}

biprog::Typed* biprog::Brackets::accept(Visitor& v) {
  type = type->accept(v);
  expr = expr->accept(v);
  return v.visit(this);
}

bool biprog::Brackets::operator<=(const Typed& o) const {
  try {
    const Brackets& o1 = dynamic_cast<const Brackets&>(o);
    return *expr <= *o1.expr && *type <= *o1.type;
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

bool biprog::Brackets::operator==(const Typed& o) const {
  try {
    const Brackets& o1 = dynamic_cast<const Brackets&>(o);
    return *expr == *o1.expr && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Brackets::output(std::ostream& out) const {
  out << '[' << *expr << ']';
}

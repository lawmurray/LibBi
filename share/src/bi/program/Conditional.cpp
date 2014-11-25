/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Conditional.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Typed* biprog::Conditional::clone() {
  return new Conditional(cond->clone(), braces->clone(), falseBraces->clone());
}

biprog::Typed* biprog::Conditional::accept(Visitor& v) {
  type = type->accept(v);
  cond = cond->accept(v);
  braces = braces->accept(v);
  falseBraces = falseBraces->accept(v);
  return v.visit(this);
}

bool biprog::Conditional::operator<=(const Typed& o) const {
  try {
    const Conditional& o1 = dynamic_cast<const Conditional&>(o);
    return *cond <= *o1.cond && *braces <= *o1.braces
        && *falseBraces <= *o1.falseBraces;
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

bool biprog::Conditional::operator==(const Typed& o) const {
  try {
    const Conditional& o1 = dynamic_cast<const Conditional&>(o);
    return *cond == *o1.cond && *braces == *o1.braces
        && *falseBraces == *o1.falseBraces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Conditional::output(std::ostream& out) const {
  out << "if " << *cond << ' ' << *braces;
  if (*falseBraces) {
    out << " else " << *falseBraces;
  }
}

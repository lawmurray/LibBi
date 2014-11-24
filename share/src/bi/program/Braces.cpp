/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirgroup.au>
 * $Rev$
 * $Date$
 */
#include "Braces.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Braces::accept(Visitor& v) {
  type = type->accept(v);
  expr = expr->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Braces::operator<=(const Typed& o) const {
  try {
    const Braces& o1 = dynamic_cast<const Braces&>(o);
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

bool biprog::Braces::operator==(const Typed& o) const {
  try {
    const Braces& o1 = dynamic_cast<const Braces&>(o);
    return *expr == *o1.expr && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Braces::output(std::ostream& out) const {
  out << '{' << *expr << '}';
}

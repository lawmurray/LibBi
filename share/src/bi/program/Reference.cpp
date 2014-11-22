/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Reference.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Reference::accept(Visitor& v) {
  type = type->accept(v);
  brackets = brackets->accept(v);
  parens = parens->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Reference::operator<=(const Typed& o) const {
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return *brackets <= *o1.brackets && *type <= *o1.type
        && *parens <= *o1.parens && *braces <= *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Reference::operator==(const Typed& o) const {
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return *brackets == *o1.brackets && *type == *o1.type
        && *parens == *o1.parens && *braces == *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Reference::output(std::ostream& out) const {
  out << name << *brackets << *parens << *braces;
}

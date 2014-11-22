/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirgroup.au>
 * $Rev$
 * $Date$
 */
#include "Group.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Group::accept(Visitor& v) {
  type = type->accept(v);
  expr = expr->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Group::operator<=(const Typed& o) const {
  try {
    const Group& o1 = dynamic_cast<const Group&>(o);
    return delim == o1.delim && *expr <= *o1.expr && *type <= *o1.type;
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

bool biprog::Group::operator==(const Typed& o) const {
  try {
    const Group& o1 = dynamic_cast<const Group&>(o);
    return delim == o1.delim && *expr == *o1.expr && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Group::output(std::ostream& out) const {
  switch (delim) {
  case BRACKETS:
    out << '[' << *expr << ']';
    break;
  case PARENS:
    out << '(' << *expr << ')';
    break;
  case BRACES:
    out << '{' << *expr << '}';
    break;
  }
}

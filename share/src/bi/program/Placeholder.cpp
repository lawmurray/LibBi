/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Placeholder.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Placeholder* biprog::Placeholder::clone() {
  return new Placeholder(name, type->clone());
}

biprog::Statement* biprog::Placeholder::acceptStatement(Visitor& v) {
  type = type->acceptStatement(v);

  return v.visitStatement(this);
}

bool biprog::Placeholder::operator<=(const Statement& o) const {
  try {
    const Placeholder& o1 = dynamic_cast<const Placeholder&>(o);
    return *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Placeholder::operator==(const Statement& o) const {
  try {
    const Placeholder& o1 = dynamic_cast<const Placeholder&>(o);
    return *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Placeholder::output(std::ostream& out) const {
  //
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Dim.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Dim* biprog::Dim::clone() {
  return new Dim(name, brackets->clone());
}

biprog::Statement* biprog::Dim::accept(Visitor& v) {
  brackets = brackets->accept(v);
  return v.visit(this);
}

bool biprog::Dim::operator<=(const Statement& o) const {
  try {
    const Dim& o1 = dynamic_cast<const Dim&>(o);
    return *brackets <= *o1.brackets;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return !*o1.brackets && !*o1.parens && !*o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Dim::operator==(const Statement& o) const {
  try {
    const Dim& o1 = dynamic_cast<const Dim&>(o);
    return *brackets == *o1.brackets;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Dim::output(std::ostream& out) const {
  out << "dim " << name << *brackets;
}

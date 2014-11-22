/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Dim.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Dim::accept(Visitor& v) {
  brackets = brackets->accept(v);
  type = type->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Dim::operator<=(const Typed& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets <= *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Dim::operator==(const Typed& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets == *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Dim::output(std::ostream& out) const {
  out << "dim " << name << *brackets;
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Type.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Type::accept(Visitor& v) {
  return v.visit(shared_from_this());
}

bool biprog::Type::operator<=(const Expression& o) const {
  return *this == o;
}

bool biprog::Type::operator==(const Expression& o) const {
  try {
    const Type& expr = dynamic_cast<const Type&>(o);
    ///@todo Avoid string comparison.
    return name.compare(expr.name) == 0;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Type::output(std::ostream& out) const {
  out << name;
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Method.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Method::accept(Visitor& v) {
  return v.visit(shared_from_this());
}

void biprog::Method::output(std::ostream& out) const {
  //
}

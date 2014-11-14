/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Function.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Function::accept(Visitor& v) {
  return v.visit(shared_from_this());
}

void biprog::Function::output(std::ostream& out) const {
  //
}

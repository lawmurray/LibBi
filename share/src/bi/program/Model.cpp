/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Model.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Model::accept(Visitor& v) {
  parens = parens->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Model::operator<=(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Model::operator==(const Expression& o) const {
  try {
    const Model& expr = dynamic_cast<const Model&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Model::output(std::ostream& out) const {
  out << "model " << name << *parens << ' ' << *braces;
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "EmptyExpression.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::EmptyExpression::accept(
    Visitor& v) {
  return v.visit(shared_from_this());
}

biprog::EmptyExpression::operator bool() const {
  return false;
}

bool biprog::EmptyExpression::operator<=(const Expression& o) const {
  return operator==(o);
}

bool biprog::EmptyExpression::operator==(const Expression& o) const {
  try {
    const EmptyExpression& expr = dynamic_cast<const EmptyExpression&>(o);
    return true;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::EmptyExpression::output(std::ostream& out) const {
  //
}

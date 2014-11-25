/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "EmptyExpression.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"
#include "../misc/compile.hpp"

#include <typeinfo>

biprog::EmptyExpression* biprog::EmptyExpression::clone() {
  return new EmptyExpression();
}

biprog::Expression* biprog::EmptyExpression::accept(Visitor& v) {
  return v.visit(this);
}

biprog::EmptyExpression::operator bool() const {
  return false;
}

bool biprog::EmptyExpression::operator<=(const Expression& o) const {
  return operator==(o);
}

bool biprog::EmptyExpression::operator==(const Expression& o) const {
  try {
    BI_UNUSED const EmptyExpression& o1 =
        dynamic_cast<const EmptyExpression&>(o);
    return true;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::EmptyExpression::output(std::ostream& out) const {
  //
}

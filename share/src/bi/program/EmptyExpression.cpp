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

biprog::Typed* biprog::EmptyExpression::clone() {
  return new EmptyExpression();
}

biprog::Typed* biprog::EmptyExpression::accept(Visitor& v) {
  return v.visit(this);
}

biprog::EmptyExpression::operator bool() const {
  return false;
}

bool biprog::EmptyExpression::operator<=(const Typed& o) const {
  return operator==(o);
}

bool biprog::EmptyExpression::operator==(const Typed& o) const {
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

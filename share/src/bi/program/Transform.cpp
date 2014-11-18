/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Transform.hpp"

#include "../visitor/Visitor.hpp"

biprog::Transform::Transform(const char* name,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope) :
    Named(name), Parenthesised(parens), Braced(braces), Scoped(scope) {
  //
}

boost::shared_ptr<biprog::Expression> biprog::Transform::accept(Visitor& v) {
  parens = parens->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Transform::operator<(const Expression& o) const {
  try {
    const Transform& expr = dynamic_cast<const Transform&>(o);
    return *parens < *expr.parens && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Transform::operator<=(const Expression& o) const {
  try {
    const Transform& expr = dynamic_cast<const Transform&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Transform::operator>(const Expression& o) const {
  try {
    const Transform& expr = dynamic_cast<const Transform&>(o);
    return *parens > *expr.parens && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Transform::operator>=(const Expression& o) const {
  try {
    const Transform& expr = dynamic_cast<const Transform&>(o);
    return *parens >= *expr.parens && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Transform::operator==(const Expression& o) const {
  try {
    const Transform& expr = dynamic_cast<const Transform&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Transform::operator!=(const Expression& o) const {
  try {
    const Transform& expr = dynamic_cast<const Transform&>(o);
    return *parens != *expr.parens || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Transform::output(std::ostream& out) const {
  out << "transform " << name << *parens << " -> " << *braces;
}

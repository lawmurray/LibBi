/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Method.hpp"

#include "../visitor/Visitor.hpp"

biprog::Method::Method(const char* name, boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope) :
    Named(name), Parenthesised(parens), Braced(braces), Scoped(scope) {
  //
}

boost::shared_ptr<biprog::Expression> biprog::Method::accept(Visitor& v) {
  parens = parens->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Method::operator<=(const Expression& o) const {
  try {
    const Method& expr = dynamic_cast<const Method&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Method::operator==(const Expression& o) const {
  try {
    const Method& expr = dynamic_cast<const Method&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Method::output(std::ostream& out) const {
  out << "method " << name << *parens << ' ' << *braces;
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Class.hpp"

#include "../visitor/Visitor.hpp"

biprog::Class::Class(const char* name, boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> base, boost::shared_ptr<Expression> braces,
    boost::shared_ptr<Scope> scope) :
    Named(name), Derived(base), Parenthesised(parens), Braced(braces), Scoped(
        scope) {
  //
}

boost::shared_ptr<biprog::Expression> biprog::Class::accept(Visitor& v) {
  parens = parens->accept(v);
  base = base->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Class::operator<=(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens <= *expr.parens && *base <= *expr.base
        && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Class::operator==(const Expression& o) const {
  try {
    const Class& expr = dynamic_cast<const Class&>(o);
    return *parens == *expr.parens && *base == *expr.base
        && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Class::output(std::ostream& out) const {
  out << "class " << name << *parens << ' ';
  if (base) {
    out << " inherits " << base;
  }
  if (*braces) {
    out << ' ' << *braces;
  }
}

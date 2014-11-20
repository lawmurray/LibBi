/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Macro.hpp"

#include "../visitor/Visitor.hpp"

biprog::Macro::Macro(const char* name, boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces, boost::shared_ptr<Expression> ret,
    boost::shared_ptr<Scope> scope) :
    Named(name), Parenthesised(parens), Braced(braces), Scoped(scope), ret(
        ret) {
  //
}

boost::shared_ptr<biprog::Expression> biprog::Macro::accept(Visitor& v) {
  parens = parens->accept(v);
  braces = braces->accept(v);
  ret = ret->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Macro::operator<=(const Expression& o) const {
  try {
    const Macro& expr = dynamic_cast<const Macro&>(o);
    return *parens <= *expr.parens && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Macro::operator==(const Expression& o) const {
  try {
    const Macro& expr = dynamic_cast<const Macro&>(o);
    return *parens == *expr.parens && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Macro::output(std::ostream& out) const {
  out << "macro " << name << *parens << ' ' << *braces << " -> " << *ret;
}

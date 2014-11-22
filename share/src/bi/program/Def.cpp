/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Def.hpp"

#include "../visitor/Visitor.hpp"

biprog::Def::Def(const char* name, boost::shared_ptr<Typed> parens,
    boost::shared_ptr<Typed> type, boost::shared_ptr<Typed> braces,
    boost::shared_ptr<Scope> scope) :
    Named(name), Parenthesised(parens), Typed(type), Braced(braces), Scoped(
        scope) {
  //
}

boost::shared_ptr<biprog::Typed> biprog::Def::accept(Visitor& v) {
  parens = parens->accept(v);
  type = type->accept(v);
  braces = braces->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Def::operator<=(const Typed& o) const {
  try {
    const Def& expr = dynamic_cast<const Def&>(o);
    return *parens <= *expr.parens && *type <= *expr.type
        && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Def::operator==(const Typed& o) const {
  try {
    const Def& expr = dynamic_cast<const Def&>(o);
    return *parens == *expr.parens && *type == *expr.type
        && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

void biprog::Def::output(std::ostream& out) const {
  out << "def " << name << *parens;
  if (type) {
    out << ':' << *type;
  }
  if (braces) {
    out << ' ' << *braces;
  }
}

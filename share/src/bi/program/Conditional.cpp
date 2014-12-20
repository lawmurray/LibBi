/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Conditional.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Conditional* biprog::Conditional::clone() {
  return new Conditional(cond->clone(), braces->clone(), falseBraces->clone());
}

biprog::Statement* biprog::Conditional::acceptStatement(Visitor& v) {
  cond = cond->acceptExpression(v);
  braces = braces->acceptExpression(v);
  falseBraces = falseBraces->acceptExpression(v);

  return v.visitStatement(this);
}

bool biprog::Conditional::operator<=(const Statement& o) const {
  try {
    const Conditional& o1 = dynamic_cast<const Conditional&>(o);
    return *cond <= *o1.cond && *braces <= *o1.braces
        && *falseBraces <= *o1.falseBraces;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return !*o1.brackets && !*o1.parens && !*o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Conditional::operator==(const Statement& o) const {
  try {
    const Conditional& o1 = dynamic_cast<const Conditional&>(o);
    return *cond == *o1.cond && *braces == *o1.braces
        && *falseBraces == *o1.falseBraces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Conditional::output(std::ostream& out) const {
  out << "if " << *cond << ' ' << *braces;
  if (*falseBraces) {
    out << " else " << *falseBraces;
  }
}

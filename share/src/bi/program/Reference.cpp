/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Reference.hpp"

#include "Def.hpp"
#include "Dim.hpp"
#include "Var.hpp"
#include "../visitor/Visitor.hpp"

#include <typeinfo>

biprog::Reference* biprog::Reference::clone() {
  return new Reference(name, brackets->clone(), parens->clone(),
      type->clone(), braces->clone(), target);
}

biprog::Expression* biprog::Reference::acceptExpression(Visitor& v) {
  type = type->acceptStatement(v);
  brackets = brackets->acceptExpression(v);
  parens = parens->acceptExpression(v);
  braces = braces->acceptExpression(v);

  return v.visitExpression(this);
}

biprog::Statement* biprog::Reference::acceptStatement(Visitor& v) {
  type = type->acceptStatement(v);
  brackets = brackets->acceptExpression(v);
  parens = parens->acceptExpression(v);
  braces = braces->acceptExpression(v);

  return v.visitStatement(this);
}

bool biprog::Reference::operator<=(const Expression& o) const {
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return *brackets <= *o1.brackets && *parens <= *o1.parens
        && *type <= *o1.type && *braces <= *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Reference::operator==(const Expression& o) const {
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return *brackets == *o1.brackets && *parens == *o1.parens
        && *type == *o1.type && *braces == *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Reference::operator<=(const Statement& o) const {
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return *brackets <= *o1.brackets && *parens <= *o1.parens
        && *type <= *o1.type && *braces <= *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Def& o1 = dynamic_cast<const Def&>(o);
    return !*brackets && *parens <= *o1.parens && !*type
        && *braces <= *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Dim& o1 = dynamic_cast<const Dim&>(o);
    return !*brackets && !*parens && !*type && !*braces;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Var& o1 = dynamic_cast<const Var&>(o);
    return *brackets <= *o1.brackets && !*parens && !*type && !*braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Reference::operator==(const Statement& o) const {
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return *brackets == *o1.brackets && *type == *o1.type
        && *parens == *o1.parens && *braces == *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Reference::output(std::ostream& out) const {
  out << name;
  if (*brackets) {
    out << *brackets;
  }
  if (*parens) {
    out << *parens;
  }
  if (*type) {
    out << ':' << *type;
  }
  if (*braces) {
    out << *braces;
  }
}

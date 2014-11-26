/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "EmptyStatement.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"
#include "../misc/compile.hpp"

#include <typeinfo>

biprog::EmptyStatement* biprog::EmptyStatement::clone() {
  return new EmptyStatement();
}

biprog::Statement* biprog::EmptyStatement::acceptStatement(Visitor& v) {
  return v.visitStatement(this);
}

biprog::EmptyStatement::operator bool() const {
  return false;
}

bool biprog::EmptyStatement::operator<=(const Statement& o) const {
  return operator==(o);
}

bool biprog::EmptyStatement::operator==(const Statement& o) const {
  try {
    BI_UNUSED const EmptyStatement& o1 =
        dynamic_cast<const EmptyStatement&>(o);
    return true;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::EmptyStatement::output(std::ostream& out) const {
  //
}

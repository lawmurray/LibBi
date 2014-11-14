/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirgroup.au>
 * $Rev$
 * $Date$
 */
#include "Group.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Group::accept(Visitor& v) {
  expr = expr->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Group::operator<(const Expression& o) const {
  try {
    const Group& group = dynamic_cast<const Group&>(o);
    return delim == group.delim && *expr < *group.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Group::operator<=(const Expression& o) const {
  try {
    const Group& group = dynamic_cast<const Group&>(o);
    return delim == group.delim && *expr <= *group.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Group::operator>(const Expression& o) const {
  try {
    const Group& group = dynamic_cast<const Group&>(o);
    return delim == group.delim && *expr > *group.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Group::operator>=(const Expression& o) const {
  try {
    const Group& group = dynamic_cast<const Group&>(o);
    return delim == group.delim && *expr >= *group.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Group::operator==(const Expression& o) const {
  try {
    const Group& group = dynamic_cast<const Group&>(o);
    return delim == group.delim && *expr == *group.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Group::operator!=(const Expression& o) const {
  try {
    const Group& group = dynamic_cast<const Group&>(o);
    return delim != group.delim || *expr != *group.expr;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Group::output(std::ostream& out) const {
  switch (delim) {
  case BRACKETS:
    out << '[' << *expr << ']';
    break;
  case PARENS:
    out << '(' << *expr << ')';
    break;
  case BRACES:
    out << '{' << *expr << '}';
    break;
  }
}

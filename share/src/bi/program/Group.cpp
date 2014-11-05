/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirgroup.au>
 * $Rev$
 * $Date$
 */
#include "Group.hpp"

#include "../misc/assert.hpp"

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
  default:
    BI_ASSERT(false);
  }
}

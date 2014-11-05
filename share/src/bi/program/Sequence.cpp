/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Sequence.hpp"

bool biprog::Sequence::operator<(const Expression& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head < *expr.head && *tail < *expr.tail;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Sequence::operator<=(const Expression& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head <= *expr.head && *tail <= *expr.tail;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Sequence::operator>(const Expression& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head > *expr.head && *tail > *expr.tail;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Sequence::operator>=(const Expression& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head >= *expr.head && *tail >= *expr.tail;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Sequence::operator==(const Expression& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head == *expr.head && *tail == *expr.tail;
  } catch (std::bad_cast e) {
    return false;
  }
}

bool biprog::Sequence::operator!=(const Expression& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head != *expr.head || *tail != *expr.tail;
  } catch (std::bad_cast e) {
    return true;
  }
}

void biprog::Sequence::output(std::ostream& out) const {
  out << *head << std::endl << *tail;
}

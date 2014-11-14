/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Sequence.hpp"

#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Expression> biprog::Sequence::accept(Visitor& v) {
  head = head->accept(v);
  tail = tail->accept(v);
  return v.visit(shared_from_this());
}

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

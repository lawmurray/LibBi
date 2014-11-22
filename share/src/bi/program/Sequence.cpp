/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Sequence.hpp"

#include "Reference.hpp"
#include "../visitor/Visitor.hpp"

boost::shared_ptr<biprog::Typed> biprog::Sequence::accept(Visitor& v) {
  type = type->accept(v);
  head = head->accept(v);
  tail = tail->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Sequence::operator<=(const Typed& o) const {
  try {
    const Sequence& o1 = dynamic_cast<const Sequence&>(o);
    return *head <= *o1.head && *tail <= *o1.tail;
  } catch (std::bad_cast e) {
    //
  }
  try {
    const Reference& o1 = dynamic_cast<const Reference&>(o);
    return !*o1.brackets && !*o1.parens && !*o1.braces && *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool biprog::Sequence::operator==(const Typed& o) const {
  try {
    const Sequence& o1 = dynamic_cast<const Sequence&>(o);
    return *head == *o1.head && *tail == *o1.tail;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

void biprog::Sequence::output(std::ostream& out) const {
  out << *head << ';' << std::endl << *tail;
}

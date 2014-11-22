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

biprog::Sequence::Sequence(boost::shared_ptr<Typed> head,
    boost::shared_ptr<Typed> tail) :
    head(head), tail(tail) {
  //
}

boost::shared_ptr<biprog::Typed> biprog::Sequence::accept(Visitor& v) {
  type = type->accept(v);
  head = head->accept(v);
  tail = tail->accept(v);
  return v.visit(shared_from_this());
}

bool biprog::Sequence::operator<=(const Typed& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head <= *expr.head && *tail <= *expr.tail;
  } catch (std::bad_cast e) {
    //
  }
//  try {
//    const Reference& expr = dynamic_cast<const Reference&>(o);
//    return type <= *expr.type && !*expr.brackets && !*expr.parens && !*expr.braces;
//  } catch (std::bad_cast e) {
//    //
//  }
  return false;
}

bool biprog::Sequence::operator==(const Typed& o) const {
  try {
    const Sequence& expr = dynamic_cast<const Sequence&>(o);
    return *head == *expr.head && *tail == *expr.tail;
  } catch (std::bad_cast e) {
    //
  }
//  try {
//    const Reference& expr = dynamic_cast<const Reference&>(o);
//    return type < *expr.type && !*expr.brackets && !*expr.parens && !*expr.braces;
//  } catch (std::bad_cast e) {
//    //
//  }
  return false;
}

void biprog::Sequence::output(std::ostream& out) const {
  out << *head << ';' << std::endl << *tail;
}

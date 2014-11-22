/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Typed.hpp"

#include "EmptyExpression.hpp"

void biprog::Typed::setType(boost::shared_ptr<Typed> type) {
  this->type = type;
}

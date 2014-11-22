/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Typed.hpp"

#include "EmptyExpression.hpp"

biprog::Typed::Typed() :
    type(boost::shared_ptr<Typed>()) {
  //
}

biprog::Typed::Typed(boost::shared_ptr<Typed> type) :
    type(type) {
  //
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Program.hpp"

#include "Named.hpp"
#include "EmptyExpression.hpp"
#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

void biprog::Program::output(std::ostream& out) const {
  out << *expr;
}

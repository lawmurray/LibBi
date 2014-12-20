/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Program.hpp"

void biprog::Program::output(std::ostream& out) const {
  out << *stmt;
}

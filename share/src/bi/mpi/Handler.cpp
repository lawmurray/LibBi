/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Handler.hpp"

#include "../misc/assert.hpp"

bool bi::Handler::done() const {
  return true;
}

void bi::Handler::join(MPI_Comm comm) {
  //
}

bool bi::Handler::canHandle(const int tag) const {
  return false;
}

void bi::Handler::handle(MPI_Comm comm, MPI_Status status) {
  BI_ASSERT(false);  // can't handle anything
}

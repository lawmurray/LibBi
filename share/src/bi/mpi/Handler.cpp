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

void bi::Handler::join(boost::mpi::communicator child) {
  //
}

bool bi::Handler::canHandle(const int tag) const {
  return false;
}

void bi::Handler::handle(boost::mpi::communicator comm,
    boost::mpi::status status) {
  /* pre-condition */
  BI_ASSERT(canHandle(status.tag()));
}

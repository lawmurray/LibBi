/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "mpi.hpp"

#ifdef ENABLE_MPI
#include "boost/mpi/communicator.hpp"
#include <mpi.h>
#endif

#include <sstream>

std::string bi::append_rank(const std::string& name) {
  #ifdef ENABLE_MPI
  boost::mpi::communicator world;
  std::stringstream stream;

  stream << name << '.' << world.rank();

  return stream.str();
  #else
  return name;
  #endif
}

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "mpi.hpp"

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

int bi::mpi_rank() {
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  return world.rank();
#else
  return 0;
#endif
}

int bi::mpi_size() {
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  return world.size();
#else
  return 1;
#endif
}

void bi::mpi_barrier() {
#ifdef ENABLE_MPI
  boost::mpi::communicator world;
  world.barrier();
#endif
}

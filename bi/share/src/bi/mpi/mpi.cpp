/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2741 $
 * $Date: 2012-06-18 16:17:26 +0800 (Mon, 18 Jun 2012) $
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

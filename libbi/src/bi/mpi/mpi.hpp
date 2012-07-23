/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2741 $
 * $Date: 2012-06-18 16:17:26 +0800 (Mon, 18 Jun 2012) $
 */
#ifndef BI_MPI_MPI_HPP
#define BI_MPI_MPI_HPP

#ifdef ENABLE_MPI
#include <mpi.h>
#include "boost/mpi/environment.hpp"
#include "boost/mpi/communicator.hpp"
#endif

#include <string>

namespace bi {
/**
 * Append rank to file name.
 *
 * @param name File name.
 */
std::string append_rank(const std::string& name);

}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_MPI_HPP
#define BI_MPI_MPI_HPP

#ifdef ENABLE_MPI
#include "boost/mpi.hpp"
#include "boost/mpi/communicator.hpp"
#include "boost/mpi/nonblocking.hpp"
#include "boost/mpi/collectives.hpp"

#include <mpi.h>
#endif

#include <string>

namespace bi {
/**
 * Append rank to file name.
 *
 * @param name File name.
 */
std::string append_rank(const std::string& name);

/**
 * Get rank.
 */
int mpi_rank();

/**
 * Get size.
 */
int mpi_size();

/**
 * Barrier.
 */
void mpi_barrier();

/**
 * Message tags.
 */
enum MPITag {
  /**
   * Server tags.
   */
  MPI_TAG_JOIN,
  MPI_TAG_DISCONNECT,

  /*
   * Stopper tags.
   */
  MPI_TAG_STOPPER_STOP,
  MPI_TAG_STOPPER_LOGWEIGHTS,

  /*
   * Adapter tags.
   */
  MPI_TAG_ADAPTER_PROPOSAL,
  MPI_TAG_ADAPTER_SAMPLES,

  /*
   * Base tag index when redistributing particles.
   */
  MPI_TAG_PARTICLE
};

}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RandomGPU.cuh"

#include "../../cuda/device.hpp"

#ifdef ENABLE_MPI
#include "../../mpi/mpi.hpp"
#endif

void bi::RandomGPU::seeds(Random& rng, const unsigned seed) {
  #ifdef ENABLE_MPI
  boost::mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();

  int s = seed*size + rank;
  #else
  int s = seed;
  #endif

  dim3 Db, Dg;
  Db.x = deviceIdealThreadsPerBlock();
  Dg.x = deviceIdealThreads()/Db.x;

  kernelSeeds<<<Dg,Db>>>(rng.devRngs, s);
  CUDA_CHECK;
}

/**
 * @file
 *
 * Documentation elements.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_INIT_HPP
#define BI_INIT_HPP

namespace bi {
/**
 * Initialise LibBi.
 *
 * @param threads Number of threads.
 */
void bi_init(const int threads = 0);
}

#include "misc/omp.hpp"
#include "ode/IntegratorConstants.hpp"

#ifdef ENABLE_CUDA
#include "cuda/cuda.hpp"
#include "cuda/device.hpp"
#endif

#ifdef ENABLE_MPI
#include "mpi/mpi.hpp"
#endif

// need to keep in same compilation unit as caller for bi_ode_init()
inline void bi::bi_init(const int threads) {
  bi_omp_init(threads);

  #ifdef ENABLE_CUDA
  #ifdef ENABLE_MPI
  boost::mpi::communicator world;
  int rank = world.rank();
  #else
  int rank = 0;
  #endif
  int dev = chooseDevice(rank);
  //std::cerr << "Rank " << rank << " using device " << dev << std::endl;
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  #endif

  bi_ode_init();
}

#endif

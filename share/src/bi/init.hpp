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

#ifdef ENABLE_SSE
#include "sse/math/scalar.hpp"
#endif

namespace bi {
/**
 * Initialise LibBi.
 *
 * @param threads Number of threads.
 */
void bi_init(const int threads = 0);

/**
 * Round up number of trajectories as required by implementation.
 *
 * @param P Minimum number of trajectories.
 *
 * @return Number of trajectories.
 *
 * The following rules are applied:
 *
 * @li for @p L on device, @p P must be either less than 32, or a
 * multiple of 32, and
 * @li for @p L on host with SSE enabled, @p P must be zero, one or a
 * multiple of four (single precision) or two (double precision).
 */
int roundup(const int P);
}

#include "misc/omp.hpp"
#include "ode/IntegratorConstants.hpp"

#ifdef ENABLE_CUDA
#include "cuda/math/magma.hpp"
#include "cuda/cuda.hpp"
#include "cuda/device.hpp"
#endif

#ifdef ENABLE_MPI
#include "boost/mpi.hpp"
#endif

// need to keep in same compilation unit as caller for bi_ode_init()
inline void bi::bi_init(const int threads) {
  bi_omp_init(threads);
  bi_ode_init();

  #ifdef ENABLE_CUDA
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  magma_init();
  #ifdef ENABLE_MPI
  boost::mpi::communicator world;
  int rank = world.rank();
  #else
  int rank = 0;
  #endif
  int dev = chooseDevice(rank);
  //std::cerr << "Rank " << rank << " using device " << dev << std::endl;
  #endif
}

inline int bi::roundup(const int P) {
  int P1 = P;

  #if defined(ENABLE_CUDA)
  /* either < 32 or a multiple of 32 number of trajectories required */
  if (P1 > 32) {
    P1 = ((P1 + 31) / 32) * 32;
  }
  #elif defined(ENABLE_SSE)
  /* zero, one or a multiple of 4 (single precision) or 2 (double
   * precision) required */
  if (P1 > 1) {
    P1 = ((P1 + BI_SSE_SIZE - 1)/BI_SSE_SIZE)*BI_SSE_SIZE;
  }
  #endif

  return P1;
}

#endif

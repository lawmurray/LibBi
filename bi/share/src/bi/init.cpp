/**
 * @file
 *
 * Documentation elements.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2700 $
 * $Date: 2012-06-12 11:59:16 +0800 (Tue, 12 Jun 2012) $
 */
#include "init.hpp"

#include "cuda/cuda.hpp"
#include "misc/omp.hpp"
#include "ode/IntegratorConstants.hpp"
#include "cuda/device.hpp"

#ifdef ENABLE_MPI
#include "boost/mpi.hpp"
#endif

void bi::bi_init(const int threads) {
  #ifdef __CUDACC__
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  #endif
  bi_omp_init(threads);
  bi_ode_init();
  #if defined(ENABLE_MPI) && defined(ENABLE_CUDA)
  boost::mpi::communicator world;
  int rank = world.rank();
  int dev = chooseDevice(rank);
  std::cerr << "Rank " << rank << " using device " << dev << std::endl;
  #endif
}

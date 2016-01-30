/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RandomHost.hpp"

#ifdef ENABLE_MPI
#include "../../mpi/mpi.hpp"
#endif

void bi::RandomHost::seeds(Random& rng, const unsigned seed) {
  #pragma omp parallel
  {
    #ifdef ENABLE_MPI
    boost::mpi::communicator world;
    const int rank = world.rank();
    const int size = world.size();

    int s = seed*size*bi_omp_max_threads + rank*bi_omp_max_threads + bi_omp_tid;
    #else
    int s = seed*bi_omp_max_threads + bi_omp_tid;
    #endif

    rng.getHostRng().seed(s);
  }
}

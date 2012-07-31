/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "RandomHost.hpp"

#ifdef ENABLE_MPI
#include "boost/mpi/communicator.hpp"
#endif

using namespace bi;

void RandomHost::seeds(Random& rng, const unsigned seed) {
  #pragma omp parallel
  {
    #ifdef ENABLE_MPI
    boost::mpi::communicator world;
    int s = seed + world.rank()*bi_omp_max_threads + bi_omp_tid;
    #else
    int s = seed + bi_omp_tid;
    #endif

    rng.getHostRng().seed(s);
  }
}

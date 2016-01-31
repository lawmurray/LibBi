/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP
#define BI_MPI_STOPPER_DISTRIBUTEDSTOPPER_HPP

#include "../../stopper/Stopper.hpp"

namespace bi {
/**
 * Distributed stopper.
 *
 * @ingroup method_stopper
 */
template<class S>
class DistributedStopper: public Stopper<S> {
public:
  /**
   * Constructor.
   *
   * @param threshold Threshold value.
   * @param maxP Maximum number of particles.
   * @param T Number of observations.
   */
  DistributedStopper(const double threshold, const int maxP, const int T);

  /**
   * Stop?
   */
  bool stop(const double maxlw = BI_INF);
};
}

#include "../mpi.hpp"

template<class S>
bi::DistributedStopper<S>::DistributedStopper(const double threshold, const int maxP,
    const int T) : Stopper<S>(threshold, maxP, T) {
  //
}

template<class S>
inline bool bi::DistributedStopper<S>::stop(const double maxlw) {
  boost::mpi::communicator world;
  const int P1 = boost::mpi::all_reduce(world, this->P, std::plus<int>());

  return P1 >= this->maxP || S::distributedStop(this->T, this->threshold, this->maxlw);
}

#endif

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
};
}

template<class S>
inline bi::DistributedStopper<S>::DistributedStopper(const double threshold, const int maxP,
    const int T) : Stopper<S>(threshold, maxP, T) {
  //
}

#endif

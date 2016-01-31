/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_STDDEVSTOPPER_HPP
#define BI_STOPPER_STDDEVSTOPPER_HPP

namespace bi {
/**
 * Stopper based on standard deviation criterion.
 *
 * @ingroup method_stopper
 */
class StdDevStopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  StdDevStopper();

  /**
   * @copydoc Stopper::stop
   */
  bool stop(const int T, const double threshold, const double maxlw = BI_INF);

#ifdef ENABLE_MPI
  /**
   * @copydoc Stopper::stop
   */
  bool distributedStop(const int T, const double threshold, const double maxlw = BI_INF);
#endif

  /**
   * @copydoc Stopper::add(const double, const double)
   */
  void add(const double lw, const double maxlw = BI_INF);

  /**
   * @copydoc Stopper::add(const V1, const double)
   */
  template<class V1>
  void add(const V1 lws, const double maxlw = BI_INF);

  /**
   * @copydoc Stopper::reset()
   */
  void reset();

private:
  /**
   * Cumulative sum.
   */
  double sum;
};
}

#include "../mpi/mpi.hpp"

inline bi::StdDevStopper::StdDevStopper() :
    sum(0.0) {
  //
}

inline bool bi::StdDevStopper::stop(const int T, const double threshold,
    const double maxlw) {
  return sum >= T * threshold;
}

#ifdef ENABLE_MPI
inline bool bi::StdDevStopper::distributedStop(const int T, const double threshold,
    const double maxlw) {
  boost::mpi::communicator world;
  double sum1 = boost::mpi::all_reduce(world, sum, std::plus<double>());

  return sum1 >= T * threshold;
}
#endif

inline void bi::StdDevStopper::add(const double lw, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(lw <= maxlw);

  double mu = bi::exp(lw);
  double s2 = bi::exp(2.0 * lw);
  double val = bi::sqrt(s2 - mu * mu);

  sum += mu / val;
}

template<class V1>
void bi::StdDevStopper::add(const V1 lws, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  double mu = sumexp_reduce(lws) / lws.size();
  double s2 = sumexpsq_reduce(lws) / lws.size();
  double val = bi::sqrt(s2 - mu * mu);

  sum += lws.size() * mu / val;
}

inline void bi::StdDevStopper::reset() {
  sum = 0.0;
}

#endif

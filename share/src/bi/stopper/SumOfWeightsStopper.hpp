/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_SUMOFWEIGHTSSTOPPER_HPP
#define BI_STOPPER_SUMOFWEIGHTSSTOPPER_HPP

#include "../math/constant.hpp"
#include "../math/function.hpp"

namespace bi {
/**
 * Stopper based on sum of weights criterion.
 *
 * @ingroup method_stopper
 */
class SumOfWeightsStopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  SumOfWeightsStopper();

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
   * @copydoc Stopper::add()
   */
  template<class V1>
  void add(const V1 lws, const double maxlw = BI_INF);

  /**
   * @copydoc Stopper::reset()
   */
  void reset();

private:
  /**
   * Sum of weights.
   */
  double sumw;
};
}

#include "../mpi/mpi.hpp"

inline bi::SumOfWeightsStopper::SumOfWeightsStopper() : sumw(0.0) {
  //
}

inline bool bi::SumOfWeightsStopper::stop(const int T,
    const double threshold, const double maxlw) {
  double minsumw = T * threshold * bi::exp(maxlw);

  return sumw >= minsumw;
}

#ifdef ENABLE_MPI
inline bool bi::SumOfWeightsStopper::distributedStop(const int T,
    const double threshold, const double maxlw) {
  boost::mpi::communicator world;
  double sumw1 = boost::mpi::all_reduce(world, sumw, std::plus<double>());
  double minsumw = T * threshold * bi::exp(maxlw);

  return sumw1 >= minsumw;
}
#endif

inline void bi::SumOfWeightsStopper::add(const double lw, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(lw <= maxlw);

  sumw += bi::exp(lw);
}

template<class V1>
void bi::SumOfWeightsStopper::add(const V1 lws, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  sumw += sumexp_reduce(lws);
}

inline void bi::SumOfWeightsStopper::reset() {
  sumw = 0.0;
}

#endif

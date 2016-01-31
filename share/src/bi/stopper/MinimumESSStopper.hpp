/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_MINIMUMESSSTOPPER_HPP
#define BI_STOPPER_MINIMUMESSSTOPPER_HPP

#include "../math/constant.hpp"
#include "../math/function.hpp"

namespace bi {
/**
 * Stopper based on ESS criterion.
 *
 * @ingroup method_stopper
 */
class MinimumESSStopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  MinimumESSStopper();

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

  /**
   * Sum of squared weights.
   */
  double sumw2;
};
}

#include "../mpi/mpi.hpp"

inline bi::MinimumESSStopper::MinimumESSStopper() :
    sumw(0.0), sumw2(0.0) {
  //
}

inline bool bi::MinimumESSStopper::stop(const int T,
    const double threshold, const double maxlw) {
  double ess = (sumw * sumw) / sumw2;
  double minsumw = bi::exp(maxlw) * (threshold - 1.0) / 2.0;

  return (sumw >= minsumw && ess >= threshold);
}

#ifdef ENABLE_MPI
inline bool bi::MinimumESSStopper::distributedStop(const int T, \
    const double threshold, const double maxlw) {
  boost::mpi::communicator world;
  double sumw1 = boost::mpi::all_reduce(world, sumw, std::plus<double>());
  double sumw21 = boost::mpi::all_reduce(world, sumw2, std::plus<double>());

  double ess = (sumw1 * sumw1) / sumw21;
  double minsumw = bi::exp(maxlw) * (threshold - 1.0) / 2.0;

  return (sumw1 >= minsumw && ess >= threshold);
}
#endif

inline void bi::MinimumESSStopper::add(const double lw, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(lw <= maxlw);

  sumw += bi::exp(lw);
  sumw2 += bi::exp(2.0*lw);
}

template<class V1>
void bi::MinimumESSStopper::add(const V1 lws, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  sumw += sumexp_reduce(lws);
  sumw2 += sumexpsq_reduce(lws);
}

inline void bi::MinimumESSStopper::reset() {
  sumw = 0.0;
  sumw2 = 0.0;
}

#endif

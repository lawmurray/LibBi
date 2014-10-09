/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_MINIMUMESSSTOPPER_HPP
#define BI_STOPPER_MINIMUMESSSTOPPER_HPP

namespace bi {
/**
 * Stopper based on ESS criterion.
 *
 * @ingroup method_stopper
 */
class MinimumESSStopper: public Stopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  MinimumESSStopper(const double threshold, const int maxP, const int T);

  /**
   * @copydoc Stopper::stop(const double maxlw)
   */
  bool stop(const double maxlw = BI_INF);

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

inline bi::MinimumESSStopper::MinimumESSStopper(const double threshold,
    const int maxP, const int T) :
    Stopper(threshold, maxP, T), sumw(0.0), sumw2(0.0) {
  //
}

inline bool bi::MinimumESSStopper::stop(const double maxlw) {
  double ess = (sumw * sumw) / sumw2;
  double miness = this->threshold;
  double minsumw = bi::exp(maxlw) * (miness - 1.0) / 2.0;

  return Stopper::stop(maxlw) || (sumw >= minsumw && ess >= miness);
}

inline void bi::MinimumESSStopper::add(const double lw, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(lw <= maxlw);

  Stopper::add(lw, maxlw);
  sumw += bi::exp(lw);
  sumw2 += bi::exp(2.0*lw);
}

template<class V1>
void bi::MinimumESSStopper::add(const V1 lws, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  Stopper::add(lws, maxlw);
  sumw += sumexp_reduce(lws);
  sumw2 += sumexpsq_reduce(lws);
}

inline void bi::MinimumESSStopper::reset() {
  Stopper::reset();
  sumw = 0.0;
  sumw2 = 0.0;
}

#endif

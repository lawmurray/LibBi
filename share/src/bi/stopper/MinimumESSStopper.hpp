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
class MinimumESSStopper : public Stopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  MinimumESSStopper(const real threshold, const int maxP, const int blockP,
      const int T);

  /**
   * @copydoc Stopper::stop()
   */
  template<class V1>
  bool stop(const V1 lws, const real maxlw);

  /**
   * @copydoc Stopper::reset()
   */
  void reset();

private:
  /**
   * Sum of weights.
   */
  real sumw;

  /**
   * Sum of squared weights.
   */
  real sumw2;
};
}

inline bi::MinimumESSStopper::MinimumESSStopper(const real threshold,
    const int maxP, const int blockP, const int T) :
    Stopper(threshold, maxP, blockP, T), sumw(0.0), sumw2(0.0) {
  //
}

template<class V1>
bool bi::MinimumESSStopper::stop(const V1 lws, const real maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  P += lws.size();
  sumw += sumexp_reduce(lws);
  sumw2 += sumexpsq_reduce(lws);

  real ess = (sumw * sumw) / sumw2;
  real miness = this->threshold;
  real minsumw = bi::exp(maxlw) * (miness - 1.0) / 2.0;

  return P >= this->maxP || (sumw >= minsumw && ess >= miness);
}

inline void bi::MinimumESSStopper::reset() {
  Stopper::reset();
  sumw = 0.0;
  sumw2 = 0.0;
}

#endif

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
class StdDevStopper : public Stopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  StdDevStopper(const real threshold, const int maxP, const int blockP,
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
   * Cumulative sum.
   */
  real sum;
};
}

inline bi::StdDevStopper::StdDevStopper(const real threshold, const int maxP,
    const int blockP, const int T) :
    Stopper(threshold, maxP, blockP, T), sum(0.0) {
  //
}

template<class V1>
bool bi::StdDevStopper::stop(const V1 lws, const real maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  real mu = sumexp_reduce(lws) / this->blockP;
  real s2 = sumexpsq_reduce(lws) / this->blockP;
  real val = bi::sqrt(s2 - mu * mu);

  sum += this->blockP * mu / val;

  real minsum = this->T * this->threshold;

  return lws.size() >= this->maxP || sum >= minsum;
}

inline void bi::StdDevStopper::reset() {
  Stopper::reset();
  sum = 0.0;
}

#endif

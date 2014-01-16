/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_VARSTOPPER_HPP
#define BI_STOPPER_VARSTOPPER_HPP

namespace bi {
/**
 * Stopper based on variance criterion.
 *
 * @ingroup method_stopper
 */
class VarStopper : public Stopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  VarStopper(const real threshold, const int maxP, const int blockP,
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

inline bi::VarStopper::VarStopper(const real threshold, const int maxP,
    const int blockP, const int T) :
    Stopper(threshold, maxP, blockP, T), sum(0.0) {
  //
}

template<class V1>
bool bi::VarStopper::stop(const V1 lws, const real maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  real mu = sumexp_reduce(lws) / this->blockP;
  real s2 = sumexpsq_reduce(lws) / this->blockP;
  real val = s2 - mu * mu;

  sum += this->blockP * mu / val;

  real minsum = this->T * this->threshold;

  return P >= this->maxP || sum >= minsum;
}

inline void bi::VarStopper::reset() {
  Stopper::reset();
  sum = 0.0;
}

#endif

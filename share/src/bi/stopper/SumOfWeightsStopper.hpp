/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_SUMOFWEIGHTSSTOPPER_HPP
#define BI_STOPPER_SUMOFWEIGHTSSTOPPER_HPP

namespace bi {
/**
 * Stopper based on sum of weights criterion.
 *
 * @ingroup method_stopper
 */
class SumOfWeightsStopper : public Stopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  SumOfWeightsStopper(const real threshold, const int maxP, const int blockP,
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
};
}

inline bi::SumOfWeightsStopper::SumOfWeightsStopper(const real threshold,
    const int maxP, const int blockP, const int T) :
    Stopper(threshold, maxP, blockP, T), sumw(0.0) {
  //
}

template<class V1>
bool bi::SumOfWeightsStopper::stop(const V1 lws, const real maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  sumw += sumexp_reduce(lws);
  real minsumw = this->T*this->threshold*bi::exp(maxlw);

  return lws.size() >= this->maxP || sumw >= minsumw;
}

inline void bi::SumOfWeightsStopper::reset() {
  Stopper::reset();
  sumw = 0.0;
}

#endif

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
class SumOfWeightsStopper: public Stopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  SumOfWeightsStopper(const double threshold, const int maxP, const int T);

  /**
   * @copydoc Stopper::stop(const double maxlw)
   */
  bool stop(const double maxlw = std::numeric_limits<double>::infinity());

  /**
   * @copydoc Stopper::add(const double, const double)
   */
  void add(const double lw, const double maxlw = std::numeric_limits<double>::infinity());

  /**
   * @copydoc Stopper::add()
   */
  template<class V1>
  void add(const V1 lws, const double maxlw = std::numeric_limits<double>::infinity());

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

inline bi::SumOfWeightsStopper::SumOfWeightsStopper(const double threshold,
    const int maxP, const int T) :
    Stopper(threshold, maxP, T), sumw(0.0) {
  //
}

inline bool bi::SumOfWeightsStopper::stop(const double maxlw) {
  double minsumw = this->T * this->threshold * bi::exp(maxlw);

  return Stopper::stop(maxlw) || sumw >= minsumw;

}

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
  Stopper::reset();
  sumw = 0.0;
}

#endif

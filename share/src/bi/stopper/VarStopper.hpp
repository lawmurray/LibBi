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
class VarStopper: public Stopper {
public:
  /**
   * @copydoc Stopper::Stopper()
   */
  VarStopper(const double threshold, const int maxP, const int T);

  /**
   * @copydoc Stopper::stop(const double maxlw)
   */
  bool stop(const double maxlw) const;

  /**
   * @copydoc Stopper::add(const double, const double)
   */
  void add(const double lw, const double maxlw);

  /**
   * @copydoc Stopper::add()
   */
  template<class V1>
  void add(const V1 lws, const double maxlw);

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

inline bi::VarStopper::VarStopper(const double threshold, const int maxP,
    const int T) :
    Stopper(threshold, maxP, T), sum(0.0) {
  //
}

inline bool bi::VarStopper::stop(const double maxlw) const {
  double minsum = this->T * this->threshold;

  return Stopper::stop(maxlw) || sum >= minsum;
}

inline void bi::VarStopper::add(const double lw, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(lw <= maxlw);

  Stopper::add(lw, maxlw);

  double mu = bi::exp(lw);
  double s2 = bi::exp(2.0 * lw);
  double val = s2 - mu * mu;

  sum += mu / val;
}

template<class V1>
void bi::VarStopper::add(const V1 lws, const double maxlw) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxlw);

  Stopper::add(lws, maxlw);

  double mu = sumexp_reduce(lws) / lws.size();
  double s2 = sumexpsq_reduce(lws) / lws.size();
  double val = s2 - mu * mu;

  sum += lws.size() * mu / val;
}

inline void bi::VarStopper::reset() {
  Stopper::reset();
  sum = 0.0;
}

#endif

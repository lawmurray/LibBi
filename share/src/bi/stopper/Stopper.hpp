/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_STOPPER_HPP
#define BI_STOPPER_STOPPER_HPP

#include "../math/scalar.hpp"

namespace bi {
/**
 * Stopper for fixed number of particles.
 *
 * @ingroup method_stopper
 *
 * Used by AdaptivePF to determine when a sufficient number of
 * particles have been propagated. Call #stop(const double maxlw) any number of times with
 * additional weights of new particles, then #reset() again before reuse.
 */
class Stopper {
public:
  /**
   * Constructor.
   *
   * @param threshold Threshold value.
   * @param maxP Maximum number of particles.
   * @param T Number of observations.
   */
  Stopper(const double threshold, const int maxP, const int T);

  /**
   * Stop?
   */
  bool stop(const double maxlw) const;

  /**
   * Add weight.
   *
   * @param lw New log-weight.
   * @param maxlw Maximum log-weight.
   */
  void add(const double lw, const double maxlw);

  /**
   * Add weights.
   *
   * @tparam V1 Vector type.
   *
   * @param lws New log-weights.
   * @param maxlw Maximum log-weight.
   */
  template<class V1>
  void add(const V1 lws, const double maxlw);

  /**
   * Reset for reuse.
   */
  void reset();

protected:
  /**
   * Threshold value.
   */
  const double threshold;

  /**
   * Maximum number of particles.
   */
  const int maxP;

  /**
   * Number of observations.
   */
  const int T;

  /**
   * Number of particles.
   */
  int P;
};
}

inline bi::Stopper::Stopper(const double threshold, const int maxP, const int T) :
    threshold(threshold), maxP(maxP), T(T), P(0) {
  //
}

inline bool bi::Stopper::stop(const double maxlw) const {
  return P >= maxP;
}

inline void bi::Stopper::add(const double lw, const double maxlw) {
  ++P;
}

template<class V1>
void bi::Stopper::add(const V1 lws, const double maxlw) {
  P += lws.size();
}

inline void bi::Stopper::reset() {
  P = 0;
}

#endif

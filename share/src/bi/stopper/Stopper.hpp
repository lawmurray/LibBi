/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_STOPPER_HPP
#define BI_STOPPER_STOPPER_HPP

#include "../math/constant.hpp"
#include "../math/function.hpp"

namespace bi {
/**
 * Stopper for fixed number of particles.
 *
 * @ingroup method_stopper
 *
 * Used by AdaptivePF to determine when a sufficient number of particles have
 * been propagated. Call #stop(const double maxlw) any number of times with
 * additional weights of new particles, then #reset() again before reuse.
 */
template<class S>
class Stopper: public S {
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
  bool stop(const double maxlw = BI_INF);

  /**
   * Add weight.
   *
   * @param lw New log-weight.
   * @param maxlw Maximum log-weight.
   */
  void add(const double lw, const double maxlw = BI_INF);

  /**
   * Add weights.
   *
   * @tparam V1 Vector type.
   *
   * @param lws New log-weights.
   * @param maxlw Maximum log-weight.
   */
  template<class V1>
  void add(const V1 lws, const double maxlw = BI_INF);

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

template<class S>
inline bi::Stopper<S>::Stopper(const double threshold, const int maxP,
    const int T) :
    threshold(threshold), maxP(maxP), T(T), P(0) {
  //
}

template<class S>
inline bool bi::Stopper<S>::stop(const double maxlw) {
  return P >= maxP || S::stop(T, threshold, maxlw);
}

template<class S>
inline void bi::Stopper<S>::add(const double lw, const double maxlw) {
  ++P;
  S::add(lw, maxlw);
}

template<class S>
template<class V1>
void bi::Stopper<S>::add(const V1 lws, const double maxlw) {
  P += lws.size();
  S::add(lws, maxlw);
}

template<class S>
inline void bi::Stopper<S>::reset() {
  P = 0;
  S::reset();
}

#endif

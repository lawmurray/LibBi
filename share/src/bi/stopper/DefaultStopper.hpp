/**
 * @file
 *
 * @author Anthony Lee
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_STOPPER_DEFAULTSTOPPER_HPP
#define BI_STOPPER_DEFAULTSTOPPER_HPP

#include "../math/constant.hpp"
#include "../math/function.hpp"

namespace bi {
/**
 * Stopper that only uses default criteria.
 *
 * @ingroup method_stopper
 */
class DefaultStopper {
public:
  /**
   * @copydoc Stopper::stop
   */
  bool stop(const int T, const double threshold, const double maxlw = BI_INF);

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
};
}

inline bool bi::DefaultStopper::stop(const int T, const double threshold,
    const double maxlw) {
  return false;
}

inline void bi::DefaultStopper::add(const double lw, const double maxlw) {
  //
}

template<class V1>
void bi::DefaultStopper::add(const V1 lws, const double maxlw) {
  //
}

inline void bi::DefaultStopper::reset() {
  //
}

#endif

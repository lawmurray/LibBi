/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_ADAPTER_WEIGHTADAPTER_HPP
#define BI_ADAPTER_WEIGHTADAPTER_HPP

#include "../misc/location.hpp"
#include "../math/loc_vector.hpp"

namespace bi {
/**
 * Adapter for weight thresholds (e.g. in MarginalSRS).
 *
 * @ingroup method_adapter
 *
 * @tparam L Location.
 */
template<Location L>
class WeightAdapter {
public:
  /**
   * Vector type.
   */
  typedef typename loc_vector<L,real>::type vector_type;

  /**
   * Constructor.
   *
   * @param initialSize Initial size of buffers (number of samples).
   */
  WeightAdapter(const int initialSize = DEFAULT_INITIAL_SIZE);

  /**
   * Add new weight.
   *
   * @tparam T1 Scalar type.
   *
   * @param lw Log-weight.
   */
  template<class T1>
  void add(const T1 lw);

  /**
   * Clear for reuse.
   */
  void clear();

protected:
  /**
   * Log-weight buffer.
   */
  vector_type lws;

  /**
   * Number of samples.
   */
  int P;

  /**
   * Initialise size of buffers.
   */
  static const int DEFAULT_INITIAL_SIZE = 256;
};
}

template<bi::Location L>
bi::WeightAdapter<L>::WeightAdapter(const int initialSize) :
    lws(initialSize), P(0) {
  //
}

template<bi::Location L>
template<class T1>
void bi::WeightAdapter<L>::add(const T1 lw) {
  if (P >= lws.size2()) {
    lws.resize(2 * lws.size(), true);
  }
  lws(P) = lw;
  ++P;
}

template<bi::Location L>
void bi::WeightAdapter<L>::clear() {
  P = 0;
}

#endif

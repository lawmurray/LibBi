/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_ADAPTER_ADAPTER_HPP
#define BI_ADAPTER_ADAPTER_HPP

#include "WeightAdapter.hpp"

namespace bi {
/**
 * Adapter for proposal distributions.
 *
 * @ingroup method_adapter
 *
 * @tparam B Model type.
 * @tparam L Location.
 */
template<class B, Location L>
class SampleAdapter: public WeightAdapter<L> {
public:
  /**
   * Matrix type.
   */
  typedef typename loc_matrix<L,real>::type matrix_type;

  /**
   * Constructor.
   *
   * @param initialSize Initial size of buffers (number of samples).
   */
  SampleAdapter(const int initialSize = WeightAdapter<L>::DEFAULT_INITIAL_SIZE);

  /**
   * Add new sample.
   *
   * @tparam V1 Vector type.
   *
   * @param x Sample.
   * @param lw Log-weight.
   */
  template<class V1>
  void add(const V1 x, const typename V1::value_type lw = 0.0);

  /**
   * Clear for reuse.
   */
  void reset();

protected:
  /**
   * Sample buffer.
   */
  matrix_type X;

  /**
   * Number of samples.
   */
  int P;
};
}

template<class B, bi::Location L>
bi::SampleAdapter<B,L>::SampleAdapter(const int initialSize) :
    X(B::NP, initialSize), P(0) {
  //
}

template<class B, bi::Location L>
template<class V1>
void bi::SampleAdapter<B,L>::add(const V1 x,
    const typename V1::value_type lw) {
  WeightAdapter<L>::add(lw);
  if (P >= X.size2()) {
    X.resize(X.size1(), 2 * X.size2(), true);
  }
  column(X, P) = x;
  ++P;
}

template<class B, bi::Location L>
void bi::SampleAdapter<B,L>::reset() {
  WeightAdapter<L>::reset();
  P = 0;
}

#endif

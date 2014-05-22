/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
#ifndef BI_ADAPTER_ADAPTER_HPP
#define BI_ADAPTER_ADAPTER_HPP

#include "../math/scalar.hpp"

namespace bi {
/**
 * Adapter for fixed number of particles.
 *
 * @ingroup method_adapter
 */
class Adapter {
public:
  /**
   * Adapt.
   *
   * @tparam V1 Vector type.
   * @tparam Q1 Pdf type.
   *
   * @param x Sample.
   * @param lw Log-weight.
   * @param[in,out] q Proposal distribution.
   *
   * Accepts a new sample and optionally adapts @p q. It is not required that
   * @p q is changed from its input value.
   */
  template<class V1, class Q1>
  void adapt(const V1 x, const typename V1::value_type lw, Q1& q);

  /**
   * Reset for reuse.
   */
  void reset();
};
}

template<class V1, class Q1>
void bi::Adapter::adapt(const V1 x, const typename V1::value_type lw, Q1& q) {

}

inline void bi::Adapter::reset() {

}

#endif

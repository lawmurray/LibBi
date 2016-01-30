/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_ADAPTER_ADAPTER_HPP
#define BI_ADAPTER_ADAPTER_HPP

#include "../model/Model.hpp"
#include "../cache/Cache2D.hpp"

namespace bi {
/**
 * Adapter.
 *
 * @ingroup method_adapter
 */
template<class A>
class Adapter: public A {
public:
  /**
   * Constructor.
   */
  Adapter(const bool local = false, const double scale = 0.25);

  /**
   * Add sample.
   *
   * @tparam V1 Vector type.
   *
   * @param s State.
   * @param lw Log-weight.
   */
  template<class S1>
  void add(const S1& s, const double lw = 0.0);

  /**
   * Add sample.
   *
   * @tparam S1 State type.
   * @tparam V1 Vector type.
   *
   * @param s State.
   * @param lws Log-weights.
   */
  template<class S1, class V1>
  void add(const S1& theta, const V1 lws);

  /**
   * Adapt.
   */
  void adapt(const int k = 0);

  /**
   * Clear adapter for reuse.
   */
  void clear();

protected:
  /**
   * Samples.
   */
  Cache2D<real> thetas;

  /**
   * Weights.
   */
  Cache2D<real> logWeights;

  /**
   * Current number of samples.
   */
  int P;
};
}

template<class A>
bi::Adapter<A>::Adapter(const bool local, const double scale) :
    A(local, scale) {
  //
}

template<class A>
template<class S1>
void bi::Adapter<A>::add(const S1& s, const double lw) {
  host_vector < real > lws(1);
  lws(0) = lw;
  add(s, lws);
}

template<class A>
template<class S1, class V1>
void bi::Adapter<A>::add(const S1& s, const V1 lws) {
  thetas.set(P, vec(s.get(P_VAR)));
  logWeights.set(P, lws);
  ++P;
}

template<class A>
void bi::Adapter<A>::adapt(const int k) {
  BOOST_AUTO(X, this->thetas.get(0, P));
  BOOST_AUTO(lws, row(this->logWeights.get(0, P), k));

  A::adapt(X, lws);
}

template<class A>
void bi::Adapter<A>::clear() {
  thetas.clear();
  logWeights.clear();
  P = 0;
}

#endif

/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_BRIDGEPF_HPP
#define BI_METHOD_BRIDGEPF_HPP

#include "LookaheadPF.hpp"
#include "../state/AuxiliaryPFState.hpp"

namespace bi {
/**
 * Bridge particle filter.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R Resampler type.
 * @tparam IO1 Output type.
 *
 * Implements the bridge particle filter as described in
 * @ref DelMoral2014 "Del Moral & Murray (2014)".
 *
 * @section Concepts
 *
 * #concept::Filter
 */
template<class B, class S, class R, class IO1>
class BridgePF: public LookaheadPF<B,S,R,IO1> {
public:
  /**
   * @copydoc BootstrapPF::BootstrapPF()
   */
  BridgePF(B& m, S* sim = NULL, R* resam = NULL, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc LookaheadPF::filter(Random&, const ScheduleIterator, const ScheduleIterator, AuxiliaryPFState<B,L>&, IO2*)
   */
  template<Location L, class IO2>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, AuxiliaryPFState<B,L>& s, IO2* inInit);

  /**
   * @copydoc LookaheadPF::filter(Random&, Schedule&, const V1, AuxiliaryPFState<B,L>&)
   */
  template<Location L, class V1>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, const V1 theta, AuxiliaryPFState<B,L>& s);

  /**
   * @copydoc LookaheadPF::step(Random&, ScheduleIterator&, const ScheduleIterator, BootstrapPFState<B,L>&)
   */
  template<bi::Location L>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      AuxiliaryPFState<B,L>& s);
//@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Update particle weights using lookahead.
   *
   * @tparam L Location.
   *
   * @param iter Current position in time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   *
   * @return Normalising constant contribution.
   */
  template<Location L>
  real bridge(Random& rng, const ScheduleIterator iter,
      const ScheduleIterator last, AuxiliaryPFState<B,L>& s);
  //@}
};

/**
 * Factory for creating BridgePF objects.
 *
 * @ingroup method
 *
 * @see BridgePF
 */
struct BridgePFFactory {
  /**
   * Create auxiliary particle filter.
   *
   * @return BridgePF object. Caller has ownership.
   *
   * @see BridgePF::BridgePF()
   */
  template<class B, class S, class R, class IO1>
  static BridgePF<B,S,R,IO1>* create(B& m, S* sim = NULL, R* resam = NULL,
      IO1* out = NULL) {
    return new BridgePF<B,S,R,IO1>(m, sim, resam, out);
  }

  /**
   * Create auxiliary particle filter.
   *
   * @return BridgePF object. Caller has ownership.
   *
   * @see BridgePF::BridgePF()
   */
  template<class B, class S, class R>
  static BridgePF<B,S,R,BootstrapPFCache<> >* create(B& m, S* sim = NULL,
      R* resam = NULL) {
    return new BridgePF<B,S,R,BootstrapPFCache<> >(m, sim, resam);
  }
};
}

#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class IO1>
bi::BridgePF<B,S,R,IO1>::BridgePF(B& m, S* sim, R* resam, IO1* out) :
    LookaheadPF<B,S,R,IO1>(m, sim, resam, out) {
  //
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class IO2>
real bi::BridgePF<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    AuxiliaryPFState<B,L>& s, IO2* inInit) {
  const int P = s.size();
  real ll = 0.0;

  ScheduleIterator iter = first;
  this->init(rng, *iter, s, inInit);
  this->output0(s);
  ll = this->correct(*iter, s);
  this->output(*iter, s);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
real bi::BridgePF<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    AuxiliaryPFState<B,L>& s) {
  // this implementation is (should be) the same as filter() above, but with
  // a different init() call

  const int P = s.size();
  real ll;

  ScheduleIterator iter = first;
  this->init(rng, theta, *iter, s);
  this->output0(s);
  ll = this->correct(*iter, s);
  this->output(*iter, s);
  while (iter + 1 != last) {
    ll += step(rng, iter, last, s);
  }
  this->term();
  this->outputT(ll);

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L>
real bi::BridgePF<B,S,R,IO1>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, AuxiliaryPFState<B,L>& s) {
  real ll = 0.0;
  do {
    ll += this->bridge(rng, iter, last, s);
    this->resample(rng, *iter, s);
    ++iter;
    this->predict(rng, *iter, s);
    ll += this->correct(*iter, s);
    this->output(*iter, s);
  } while (iter + 1 != last && !iter->isObserved());

  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L>
real bi::BridgePF<B,S,R,IO1>::bridge(Random& rng, const ScheduleIterator iter,
    const ScheduleIterator last, AuxiliaryPFState<B,L>& s) {
  real ll = 0.0;
  if (iter->hasBridge() && !iter->isObserved()
      && last->indexObs() > iter->indexObs()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();

    this->m.bridgeLogDensities(s,
        this->getSim()->getObs()->getMask(iter->indexObs()),
        s.logAuxWeights());

    axpy(1.0, s.logAuxWeights(), s.logWeights());
    ll = logsumexp_reduce(s.logWeights())
        - bi::log(static_cast<real>(s.size()));
  }
  return ll;
}

#endif

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
 *
 * Implements the bridge particle filter as described in
 * @ref DelMoral2014 "Del Moral & Murray (2014)".
 */
template<class B, class S, class R>
class BridgePF: public LookaheadPF<B,S,R> {
public:
  /**
   * @copydoc BootstrapPF::BootstrapPF()
   */
  BridgePF(B& m, S& sim, R& resam);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc LookaheadPF::step()
   */
  template<bi::Location L, class IO1>
  real step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      AuxiliaryPFState<B,L>& s, IO1* out);
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
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R>
bi::BridgePF<B,S,R>::BridgePF(B& m, S& sim, R& resam) :
    LookaheadPF<B,S,R>(m, sim, resam) {
  //
}

template<class B, class S, class R>
template<bi::Location L, class IO1>
real bi::BridgePF<B,S,R>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, AuxiliaryPFState<B,L>& s, IO1* out) {
  real ll = 0.0;
  do {
    ll += this->bridge(rng, iter, last, s);
    this->resample(rng, *iter, s);
    ++iter;
    this->predict(rng, *iter, s);
    ll += this->correct(rng, *iter, s);
    this->output(*iter, s, out);
  } while (iter + 1 != last && !iter->isObserved());

  return ll;
}

template<class B, class S, class R>
template<bi::Location L>
real bi::BridgePF<B,S,R>::bridge(Random& rng, const ScheduleIterator iter,
    const ScheduleIterator last, AuxiliaryPFState<B,L>& s) {
  real ll = 0.0;
  if (iter->hasBridge() && !iter->isObserved()
      && last->indexObs() > iter->indexObs()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();

    this->m.bridgeLogDensities(s,
        this->sim.obs.getMask(iter->indexObs()), s.logAuxWeights());

    axpy(1.0, s.logAuxWeights(), s.logWeights());
    ll = logsumexp_reduce(s.logWeights())
        - bi::log(static_cast<real>(s.size()));
  }
  return ll;
}

#endif

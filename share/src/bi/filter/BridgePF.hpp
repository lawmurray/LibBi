/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_FILTER_BRIDGEPF_HPP
#define BI_FILTER_BRIDGEPF_HPP

#include "BootstrapPF.hpp"
#include "../state/AuxiliaryPFState.hpp"

namespace bi {
/**
 * Bridge particle filter.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam F Forcer type.
 * @tparam O Observer type.
 * @tparam R Resampler type.
 *
 * Implements the bridge particle filter as described in
 * @ref DelMoral2014 "Del Moral & Murray (2014)".
 */
template<class B, class F, class O, class R>
class BridgePF: public BootstrapPF<B,F,O,R> {
public:
  /**
   * @copydoc BootstrapPF::BootstrapPF()
   */
  BridgePF(B& m, F& in, O& obs, R& resam);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BridgePF::step()
   */
  template<class S1, class IO1>
  void step(Random& rng, ScheduleIterator& iter, const ScheduleIterator last,
      S1& s, IO1& out);
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Update particle weights using lookahead.
   *
   * @tparam S1 State type.
   *
   * @param iter Current position in time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void bridge(Random& rng, const ScheduleIterator iter,
      const ScheduleIterator last, S1& s);

  /**
   * @copydoc BootstrapPF::correct()
   */
  template<class S1>
  void correct(Random& rng, const ScheduleElement now, S1& s);
//@}

protected:
  /**
   * Compute the maximum log-weight of a particle at the current time under
   * the bridge density.
   *
   * @tparam S1 State type.
   *
   * @param s State.
   *
   * @return Maximum log-weight.
   */
  template<class S1>
  double getMaxLogWeightBridge(const ScheduleElement now, S1& s);
};
}

#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class F, class O, class R>
bi::BridgePF<B,F,O,R>::BridgePF(B& m, F& in, O& obs, R& resam) :
    BootstrapPF<B,F,O,R>(m, in, obs, resam) {
  //
}

template<class B, class F, class O, class R>
template<class S1, class IO1>
void bi::BridgePF<B,F,O,R>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, S1& s, IO1& out) {
  do {
    this->bridge(rng, iter, last, s);
    this->resample(rng, *iter, s);
    ++iter;
    this->predict(rng, *iter, s);
    this->correct(rng, *iter, s);
    this->output(*iter, s, out);
  } while (iter + 1 != last && !iter->isObserved());
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BridgePF<B,F,O,R>::bridge(Random& rng, const ScheduleIterator iter,
    const ScheduleIterator last, S1& s) {
  if (iter->hasBridge() && !iter->isObserved()
      && last->indexObs() > iter->indexObs()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();

    this->m.bridgeLogDensities(s, this->obs.getMask(iter->indexObs()),
        s.logAuxWeights());

    axpy(1.0, s.logAuxWeights(), s.logWeights());

    double lW;
    s.ess = this->resam.reduce(s.logWeights(), &lW);
    s.logIncrements(iter->indexObs()) = lW - s.logLikelihood;
    s.logLikelihood = lW;
  }
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BridgePF<B,F,O,R>::correct(Random& rng, const ScheduleElement now,
    S1& s) {
  if (now.isObserved()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();
    BootstrapPF<B,F,O,R>::correct(rng, now, s);
  }
}

template<class B, class F, class O, class R>
template<class S1>
double bi::BridgePF<B,F,O,R>::getMaxLogWeightBridge(const ScheduleElement now,
    S1& s) {
  return this->m.bridgeMaxLogDensity(s, this->obs.getMask(now.indexObs()));
}

#endif

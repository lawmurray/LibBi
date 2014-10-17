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

  /**
   * @copydoc BootstrapPF::resample()
   */
  template<class S1>
  void resample(Random& rng, const ScheduleElement now, S1& s);
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
  }
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BridgePF<B,F,O,R>::correct(Random& rng, const ScheduleElement now,
    S1& s) {
  if (now.isObserved()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();
    this->m.observationLogDensities(s, this->obs.getMask(now.indexObs()),
        s.logWeights());
  }
}

template<class B, class F, class O, class R>
template<class S1>
void bi::BridgePF<B,F,O,R>::resample(Random& rng, const ScheduleElement now,
    S1& s) {
  if (this->resam.isTriggered(now, s.logWeights(), &s.logLikelihood)) {
    if (resampler_needs_max<R>::value) {
      if (now.isObserved()) {
        this->resam.setMaxLogWeight(this->getMaxLogWeight(now, s));
      } else if (now.hasBridge()) {
        this->resam.setMaxLogWeight(this->getMaxLogWeightBridge(now, s));
      }
    }
    typename precompute_type<R,S1::location>::type pre;
    this->resam.precompute(s.logWeights(), pre);
    if (now.hasOutput()) {
      this->resam.ancestorsPermute(rng, s.logWeights(), s.ancestors(), pre);
      this->resam.copy(s.ancestors(), s.getDyn());
    } else {
      typename S1::temp_int_vector_type as1(s.ancestors().size());
      this->resam.ancestorsPermute(rng, s.logWeights(), as1, pre);
      this->resam.copy(as1, s.getDyn());
      bi::gather(as1, s.ancestors(), s.ancestors());
      bi::gather(as1, s.logAuxWeights(), s.logAuxWeights());
    }
    s.logWeights().clear();
  } else if (now.hasOutput()) {
    seq_elements(s.ancestors(), 0);
  }
}

template<class B, class F, class O, class R>
template<class S1>
double bi::BridgePF<B,F,O,R>::getMaxLogWeightBridge(const ScheduleElement now,
    S1& s) {
  return this->m.bridgeMaxLogDensity(s, this->obs.getMask(now.indexObs()));
}

#endif

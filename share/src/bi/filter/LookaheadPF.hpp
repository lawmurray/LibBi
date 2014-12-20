/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_FILTER_LOOKAHEADPF_HPP
#define BI_FILTER_LOOKAHEADPF_HPP

#include "BridgePF.hpp"
#include "../state/AuxiliaryPFState.hpp"

namespace bi {
/**
 * Single-pilot lookahead particle filter.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam F Forcer type.
 * @tparam O Observer type.
 * @tparam R Resampler type.
 */
template<class B, class F, class O, class R>
class LookaheadPF: public BridgePF<B,F,O,R> {
public:
  /**
   * @copydoc BridgePF::BridgePF()
   */
  LookaheadPF(B& m, F& in, O& obs, R& resam);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BootstrapPF::step()
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
   * Perform lookahead.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] rng Random number generator.
   * @param iter Current position in time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   */
  template<class S1>
  void bridge(Random& rng, const ScheduleIterator iter,
      const ScheduleIterator last, S1& s);
  //@}
};
}

#include "../math/loc_temp_matrix.hpp"

template<class B, class F, class O, class R>
bi::LookaheadPF<B,F,O,R>::LookaheadPF(B& m, F& in, O& obs, R& resam) :
    BridgePF<B,F,O,R>(m, in, obs, resam) {
  //
}

template<class B, class F, class O, class R>
template<class S1, class IO1>
void bi::LookaheadPF<B,F,O,R>::step(Random& rng, ScheduleIterator& iter,
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
void bi::LookaheadPF<B,F,O,R>::bridge(Random& rng,
    const ScheduleIterator iter, const ScheduleIterator last, S1& s) {
  if (iter->hasBridge() && last->indexObs() > iter->indexObs()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();

    /* save previous state */
    typename loc_temp_matrix<S1::location,real>::type X(s.getDyn().size1(),
        s.getDyn().size2());
    X = s.getDyn();
    real t = s.getTime();
    real tInput = s.getLastInputTime();
    real tObs = s.getNextObsTime();

    /* lookahead */
    ScheduleIterator iter1 = iter;
    do {
      ++iter1;
      this->lookahead(rng, *iter1, s);
    } while (!iter1->isObserved());
    this->m.lookaheadObservationLogDensities(s,
        this->obs.getMask(iter1->indexObs()), s.logAuxWeights());

    /* restore previous state */
    s.getDyn() = X;
    s.setTime(t);
    s.setLastInputTime(tInput);
    s.setNextObsTime(tObs);

    axpy(1.0, s.logAuxWeights(), s.logWeights());
  }
}

#endif

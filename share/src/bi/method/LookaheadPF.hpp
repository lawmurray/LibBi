/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_LOOKAHEADPF_HPP
#define BI_METHOD_LOOKAHEADPF_HPP

#include "BootstrapPF.hpp"
#include "../state/AuxiliaryPFState.hpp"

namespace bi {
/**
 * Auxiliary particle filter with lookahead.
 *
 * @ingroup method_filter
 *
 * @tparam B Model type.
 * @tparam S Simulator type.
 * @tparam R Resampler type.
 */
template<class B, class S, class R>
class LookaheadPF: public BootstrapPF<B,S,R> {
public:
  /**
   * @copydoc BootstrapPF::BootstrapPF()
   */
  LookaheadPF(B& m, S& sim, R& resam);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BootstrapPF::step()
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
   * @copydoc BootstrapPF::init()
   */
  template<Location L, class IO1, class IO2>
  void init(Random& rng, const ScheduleElement now, AuxiliaryPFState<B,L>& s,
      IO1* out, IO2* inInit);

  /**
   * @copydoc BootstrapPF::init()
   */
  template<Location L, class V1, class IO1>
  void init(Random& rng, const V1 theta, const ScheduleElement now,
      AuxiliaryPFState<B,L>& s, IO1* out);

  /**
   * Perform lookahead.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param iter Current position in time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   *
   * @return Normalising constant contribution.
   */
  template<Location L>
  real lookahead(Random& rng, const ScheduleIterator iter,
      const ScheduleIterator last, AuxiliaryPFState<B,L>& s);

  /**
   * @copydoc BootstrapPF::correct()
   */
  template<Location L>
  real correct(Random& rng, const ScheduleElement now, AuxiliaryPFState<B,L>& s);

  /**
   * @copydoc BootstrapPF::resample()
   */
  template<Location L>
  bool resample(Random& rng, const ScheduleElement now,
      AuxiliaryPFState<B,L>& s);
  //@}

protected:
  /**
   * Compute the maximum log-weight of a particle at the current time under
   * the bridge density.
   *
   * @tparam L Location.
   *
   * @param s State.
   *
   * @return Maximum log-weight.
   */
  template<Location L>
  real getMaxLogWeightBridge(const ScheduleElement now,
      AuxiliaryPFState<B,L>& s);

};
}

#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R>
bi::LookaheadPF<B,S,R>::LookaheadPF(B& m, S& sim, R& resam) :
    BootstrapPF<B,S,R>(m, sim, resam) {
  //
}

template<class B, class S, class R>
template<bi::Location L, class IO1, class IO2>
void bi::LookaheadPF<B,S,R>::init(Random& rng, const ScheduleElement now,
    AuxiliaryPFState<B,L>& s, IO1* out, IO2* inInit) {
  BootstrapPF<B,S,R>::init(rng, now, s, out, inInit);
  s.logAuxWeights().clear();
}

template<class B, class S, class R>
template<bi::Location L, class V1, class IO1>
void bi::LookaheadPF<B,S,R>::init(Random& rng, const V1 theta,
    const ScheduleElement now, AuxiliaryPFState<B,L>& s, IO1* out) {
  BootstrapPF<B,S,R>::init(rng, theta, now, s, out);
  s.logAuxWeights().clear();
}

template<class B, class S, class R>
template<bi::Location L, class IO1>
real bi::LookaheadPF<B,S,R>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, AuxiliaryPFState<B,L>& s, IO1* out) {
  real ll = 0.0;
  do {
    ll += this->lookahead(rng, iter, last, s);
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
real bi::LookaheadPF<B,S,R>::lookahead(Random& rng,
    const ScheduleIterator iter, const ScheduleIterator last,
    AuxiliaryPFState<B,L>& s) {
  real ll = 0.0;
  if (iter->hasBridge() && last->indexObs() > iter->indexObs()) {
    if (iter->isObserved()) {
      Resampler::normalise(s.logWeights());
    }
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();

    /* save previous state */
    typename loc_temp_matrix<L,real>::type X(s.getDyn().size1(),
        s.getDyn().size2());
    X = s.getDyn();
    real t = s.getTime();
    real tInput = s.getLastInputTime();
    real tObs = s.getNextObsTime();

    /* lookahead */
    ScheduleIterator iter1 = iter;
    do {
      ++iter1;
      this->sim.lookahead(rng, *iter1, s);
    } while (!iter1->isObserved());
    this->m.lookaheadObservationLogDensities(s,
        this->sim.obs.getMask(iter1->indexObs()), s.logAuxWeights());

    /* restore previous state */
    s.getDyn() = X;
    s.setTime(t);
    s.setLastInputTime(tInput);
    s.setNextObsTime(tObs);

    axpy(1.0, s.logAuxWeights(), s.logWeights());
    ll = logsumexp_reduce(s.logWeights())
        - bi::log(static_cast<real>(s.size()));
  }
  return ll;
}

template<class B, class S, class R>
template<bi::Location L>
real bi::LookaheadPF<B,S,R>::correct(Random& rng, const ScheduleElement now,
    AuxiliaryPFState<B,L>& s) {
  real ll = 0.0;
  if (now.isObserved()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();

    this->m.observationLogDensities(s,
        this->sim.obs.getMask(now.indexObs()), s.logWeights());

    ll = logsumexp_reduce(s.logWeights())
        - bi::log(static_cast<real>(s.size()));
  }
  return ll;
}

template<class B, class S, class R>
template<bi::Location L>
bool bi::LookaheadPF<B,S,R>::resample(Random& rng, const ScheduleElement now,
    AuxiliaryPFState<B,L>& s) {
  bool r = false;
  if (now.isObserved()) {
    r = this->resam.isTriggered(s.logWeights());
    if (r) {
      if (resampler_needs_max<R>::value) {
        this->resam.setMaxLogWeight(this->getMaxLogWeight(now, s));
      }
      this->resam.resample(rng, s.logWeights(), s.ancestors(), s.getDyn());
      bi::gather(s.ancestors(), s.logAuxWeights(), s.logAuxWeights());
    } else {
      seq_elements(s.ancestors(), 0);
      Resampler::normalise(s.logWeights());
    }
  } else if (now.hasBridge()) {
    r = this->resam.isTriggeredBridge(s.logWeights());
    if (r) {
      if (resampler_needs_max<R>::value) {
        this->resam.setMaxLogWeight(this->getMaxLogWeightBridge(now, s));
      }
      if (now.hasOutput()) {
        this->resam.resample(rng, s.logWeights(), s.ancestors(), s.getDyn());
        bi::gather(s.ancestors(), s.logAuxWeights(), s.logAuxWeights());
      } else {
        typename State<B,L>::int_vector_type as1(s.ancestors().size());
        this->resam.resample(rng, s.logWeights(), s.ancestors(), s.getDyn());
        bi::gather(as1, s.logAuxWeights(), s.logAuxWeights());
        bi::gather(as1, s.ancestors(), s.ancestors());
      }
    } else {
      if (now.hasOutput()) {
        seq_elements(s.ancestors(), 0);
      }
      Resampler::normalise(s.logWeights());
    }
  } else if (now.hasOutput()) {
    seq_elements(s.ancestors(), 0);
  }
  return r;
}

template<class B, class S, class R>
template<bi::Location L>
real bi::LookaheadPF<B,S,R>::getMaxLogWeightBridge(const ScheduleElement now,
    AuxiliaryPFState<B,L>& s) {
  return this->m.bridgeMaxLogDensity(s,
      this->sim.obs.getMask(now.indexObs()));
}

#endif

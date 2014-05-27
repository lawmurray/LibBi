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
 * @tparam IO1 Output type.
 *
 * @section Concepts
 *
 * #concept::Filter
 */
template<class B, class S, class R, class IO1>
class LookaheadPF: public BootstrapPF<B,S,R,IO1> {
public:
  /**
   * @copydoc BootstrapPF::BootstrapPF()
   */
  LookaheadPF(B& m, S* sim = NULL, R* resam = NULL, IO1* out = NULL);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * @copydoc BootstrapPF::filter(Random&, const ScheduleIterator, const ScheduleIterator, BootstrapPFState<B,L>&, IO2*)
   */
  template<Location L, class IO2>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, AuxiliaryPFState<B,L>& s, IO2* inInit);

  /**
   * @copydoc BootstrapPF::filter(Random&, Schedule&, const V1, BootstrapPFState<B,L>&)
   */
  template<Location L, class V1>
  real filter(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, const V1 theta, AuxiliaryPFState<B,L>& s);

  /**
   * @copydoc BootstrapPF::step(Random&, ScheduleIterator&, const ScheduleIterator, BootstrapPFState<B,L>&)
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
   * Initialise.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[out] s State.
   * @param inInit Initialisation file.
   */
  template<Location L, class IO2>
  void init(Random& rng, const ScheduleElement now, AuxiliaryPFState<B,L>& s,
      IO2* inInit);

  /**
   * Initialise, with fixed parameters and starting at time zero.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param theta Parameters.
   * @param[out] s State.
   */
  template<Location L, class V1>
  void init(Random& rng, const V1 theta, const ScheduleElement now,
      AuxiliaryPFState<B,L>& s);

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
   * Update particle weights using observations at the current time.
   *
   * @tparam L Location.
   *
   * @param now Current step in time schedule.
   * @param s State.
   *
   * @return Estimate of the incremental log-likelihood.
   */
  template<Location L>
  real correct(const ScheduleElement now, AuxiliaryPFState<B,L>& s);

  /**
   * Resample after bridge weighting.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param now Current step in time schedule.
   * @param[in,out] s State.
   *
   * @return True if resampling was performed, false otherwise.
   */
  template<Location L>
  bool resample(Random& rng, const ScheduleElement now,
      AuxiliaryPFState<B,L>& ss);
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

/**
 * Factory for creating LookaheadPF objects.
 *
 * @ingroup method
 *
 * @see LookaheadPF
 */
struct LookaheadPFFactory {
  /**
   * Create auxiliary particle filter.
   *
   * @return LookaheadPF object. Caller has ownership.
   *
   * @see LookaheadPF::LookaheadPF()
   */
  template<class B, class S, class R, class IO1>
  static LookaheadPF<B,S,R,IO1>* create(B& m, S* sim = NULL, R* resam = NULL,
      IO1* out = NULL) {
    return new LookaheadPF<B,S,R,IO1>(m, sim, resam, out);
  }

  /**
   * Create auxiliary particle filter.
   *
   * @return LookaheadPF object. Caller has ownership.
   *
   * @see LookaheadPF::LookaheadPF()
   */
  template<class B, class S, class R>
  static LookaheadPF<B,S,R,BootstrapPFCache<> >* create(B& m, S* sim = NULL,
      R* resam = NULL) {
    return new LookaheadPF<B,S,R,BootstrapPFCache<> >(m, sim, resam);
  }
};
}

#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, class S, class R, class IO1>
bi::LookaheadPF<B,S,R,IO1>::LookaheadPF(B& m, S* sim, R* resam, IO1* out) :
    BootstrapPF<B,S,R,IO1>(m, sim, resam, out) {
  //
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class IO2>
real bi::LookaheadPF<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last,
    AuxiliaryPFState<B,L>& s, IO2* inInit) {
  const int P = s.size();
  real ll;

  ScheduleIterator iter = first;
  init(rng, *iter, s, inInit);
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
real bi::LookaheadPF<B,S,R,IO1>::filter(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, const V1 theta,
    AuxiliaryPFState<B,L>& s) {
  // this implementation is (should be) the same as filter() above, but with
  // a different init() call

  const int P = s.size();
  real ll;

  ScheduleIterator iter = first;
  init(rng, theta, *iter, s);
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
template<bi::Location L, class IO2>
void bi::LookaheadPF<B,S,R,IO1>::init(Random& rng, const ScheduleElement now,
    AuxiliaryPFState<B,L>& s, IO2* inInit) {
  BootstrapPF<B,S,R,IO1>::init(rng, now, s, inInit);
  s.logAuxWeights().clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L, class V1>
void bi::LookaheadPF<B,S,R,IO1>::init(Random& rng, const V1 theta,
    const ScheduleElement now, AuxiliaryPFState<B,L>& s) {
  BootstrapPF<B,S,R,IO1>::init(rng, theta, now, s);
  s.logAuxWeights().clear();
}

template<class B, class S, class R, class IO1>
template<bi::Location L>
real bi::LookaheadPF<B,S,R,IO1>::step(Random& rng, ScheduleIterator& iter,
    const ScheduleIterator last, AuxiliaryPFState<B,L>& s) {
  real ll = 0.0;
  do {
    ll += this->lookahead(rng, iter, last, s);
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
real bi::LookaheadPF<B,S,R,IO1>::lookahead(Random& rng,
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
      this->getSim()->lookahead(rng, *iter1, s);
    } while (!iter1->isObserved());
    this->m.lookaheadObservationLogDensities(s,
        this->getSim()->getObs()->getMask(iter1->indexObs()),
        s.logAuxWeights());

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

template<class B, class S, class R, class IO1>
template<bi::Location L>
real bi::LookaheadPF<B,S,R,IO1>::correct(const ScheduleElement now,
    AuxiliaryPFState<B,L>& s) {
  real ll = 0.0;
  if (now.isObserved()) {
    axpy(-1.0, s.logAuxWeights(), s.logWeights());
    s.logAuxWeights().clear();

    this->m.observationLogDensities(s,
        this->getSim()->getObs()->getMask(now.indexObs()), s.logWeights());

    ll = logsumexp_reduce(s.logWeights())
        - bi::log(static_cast<real>(s.size()));
  }
  return ll;
}

template<class B, class S, class R, class IO1>
template<bi::Location L>
bool bi::LookaheadPF<B,S,R,IO1>::resample(Random& rng,
    const ScheduleElement now, AuxiliaryPFState<B,L>& s) {
  bool r = false;
  if (now.isObserved()) {
    r = this->resam != NULL && this->resam->isTriggered(s.logWeights());
    if (r) {
      if (resampler_needs_max<R>::value) {
        this->resam->setMaxLogWeight(this->getMaxLogWeight(now, s));
      }
      this->resam->resample(rng, s.logWeights(), s.ancestors(), s.getDyn());
      bi::gather(s.ancestors(), s.logAuxWeights(), s.logAuxWeights());
    } else {
      seq_elements(s.ancestors(), 0);
      Resampler::normalise(s.logWeights());
    }
  } else if (now.hasBridge()) {
    r = this->resam != NULL && this->resam->isTriggeredBridge(s.logWeights());
    if (r) {
      if (resampler_needs_max<R>::value) {
        this->resam->setMaxLogWeight(this->getMaxLogWeightBridge(now, s));
      }
      if (now.hasOutput()) {
        this->resam->resample(rng, s.logWeights(), s.ancestors(), s.getDyn());
        bi::gather(s.ancestors(), s.logAuxWeights(), s.logAuxWeights());
      } else {
        typename State<B,L>::int_vector_type as1(s.ancestors().size());
        this->resam->resample(rng, s.logWeights(), s.ancestors(), s.getDyn());
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

template<class B, class S, class R, class IO1>
template<bi::Location L>
real bi::LookaheadPF<B,S,R,IO1>::getMaxLogWeightBridge(
    const ScheduleElement now, AuxiliaryPFState<B,L>& s) {
  return this->m.bridgeMaxLogDensity(s,
      this->sim->getObs()->getMask(now.indexObs()));
}

#endif
